import torch
import torch as th
from torch import nn
import torch.nn.functional as f
import numpy as np
import gymnasium
from hrl_nn_mlp import high_level_policy, low_level_policy, FFNN
from torch.distributions import MultivariateNormal, Categorical
from torch.optim import Adam

class HiPPO:
    def __init__(self, env):
        self.env = env

        # Determine observation space dimensions
        if isinstance(env.observation_space, gymnasium.spaces.Discrete):
            self.obs_dim = env.observation_space.n
            self.is_discrete_obs = True
        else:
            self.obs_dim = env.observation_space.shape[0]
            self.is_discrete_obs = False

        # Determine action space dimensions
        if isinstance(env.action_space, gymnasium.spaces.Discrete):
            self.action_dim = env.action_space.n
            self.is_discrete = True
        else:
            self.action_dim = env.action_space.shape[0]
            self.is_discrete = False

        # Adjust network input dimensions based on observation type
        hl_input_dim = self.obs_dim if not self.is_discrete_obs else self.obs_dim
        ll_input_dim = hl_input_dim + (self.action_dim if not self.is_discrete else self.action_dim)

        self.high_level_policy = high_level_policy(hl_input_dim, self.action_dim)
        self.low_level_policy = low_level_policy(ll_input_dim, self.action_dim)
        self.low_level_critic = FFNN(ll_input_dim, 1)
        self.high_level_critic = FFNN(hl_input_dim, 1)
        self._init_hyperparameters()
        self.hl_optim = Adam(self.high_level_policy.parameters(), lr=self.hl_lr)
        self.ll_optim = Adam(self.low_level_policy.parameters(), lr=self.ll_lr)
        self.ll_critic_optim = Adam(self.low_level_critic.parameters(), lr=self.ll_lr)
        self.hl_critic_optim = Adam(self.high_level_critic.parameters(), lr=self.hl_lr)
        if not self.is_discrete:
            self.covariant_var = torch.full(size=(self.action_dim,), fill_value=0.5)
            self.covariant_matrix = torch.diag(self.covariant_var)
            self.ll_covariant_var = torch.full(size=(self.action_dim,), fill_value=0.5)
            self.ll_covariant_matrix = torch.diag(self.ll_covariant_var)
        else:
            self.covariant_matrix = None
            self.ll_covariant_matrix = None

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 6400
        self.max_timesteps_per_low_level_episode = 200
        self.max_timesteps_per_high_level_episode = 1600
        self.low_level_steps_per_high_level_steps = 10
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.hl_lr = 0.005
        self.ll_lr = 0.005

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_lens = []

        hl_obs = []
        hl_acts = []
        hl_log_probs = []
        hl_rewards = []
        hl_batch_lens = []
        hl_batch_rewards = []

        t = 0
        while t < self.timesteps_per_batch:
            obs, _ = self.env.reset()
            obs = self.get_high_level_obs(obs)
            done = False
            ep_rewards = []
            hl_ep_rewards = []
            ep_t = 0

            while not done and ep_t < self.max_timesteps_per_high_level_episode:
                hl_obs.append(obs.clone())
                hl_action, hl_log_prob = self.get_mean_high_level_action(obs)
                hl_acts.append(hl_action.item() if self.is_discrete else hl_action.detach().numpy())
                hl_log_probs.append(hl_log_prob.item())
                low_level_rewards = []
                low_level_steps = 0

                for _ in range(self.low_level_steps_per_high_level_steps):
                    combined_obs = self.combine_obs_subgoal(obs, hl_action)
                    ll_action, ll_log_prob = self.get_mean_low_level_action(combined_obs)
                    action_env = ll_action.item() if self.is_discrete else ll_action.numpy()
                    next_obs, reward, terminated, truncated, _ = self.env.step(action_env)
                    done = terminated or truncated

                    batch_obs.append(combined_obs.numpy())
                    batch_acts.append(ll_action.item() if self.is_discrete else ll_action.detach().numpy())
                    batch_log_probs.append(ll_log_prob.item())
                    ep_rewards.append(reward)
                    low_level_rewards.append(reward)
                    obs = self.get_high_level_obs(next_obs)
                    t += 1
                    low_level_steps += 1
                    ep_t += 1

                    if done or t >= self.timesteps_per_batch:
                        break

                hl_reward = sum(low_level_rewards)
                hl_ep_rewards.append(hl_reward)

                if done or t >= self.timesteps_per_batch:
                    break

            batch_lens.append(ep_t)
            batch_rewards.append(ep_rewards)
            hl_batch_lens.append(len(hl_ep_rewards))
            hl_batch_rewards.append(hl_ep_rewards)

        # Convert lists to tensors
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float if not self.is_discrete else torch.long)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rew_to_go = self.compute_rtgs(batch_rewards)
        hl_obs = torch.tensor(np.array([o.numpy() for o in hl_obs]), dtype=torch.float)
        hl_acts = torch.tensor(hl_acts, dtype=torch.float if not self.is_discrete else torch.long)
        hl_log_probs = torch.tensor(hl_log_probs, dtype=torch.float)
        hl_rew_to_go = self.compute_rtgs(hl_batch_rewards)

        return batch_obs, batch_acts, batch_log_probs, batch_rew_to_go, batch_lens, hl_obs, hl_acts, hl_log_probs, hl_rew_to_go, hl_batch_lens

    def compute_rtgs(self, batch_rewards):
        batch_rtgs = []
        for ep_rewards in batch_rewards:
            discounted_reward = 0
            rtgs = []
            for reward in reversed(ep_rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                rtgs.insert(0, discounted_reward)
            batch_rtgs.extend(rtgs)
        return torch.tensor(batch_rtgs, dtype=torch.float)

    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:

            (batch_obs, batch_acts, batch_log_probs, batch_rew_to_go, batch_lens,
             hl_obs, hl_acts, hl_log_probs, hl_rew_to_go, hl_lens) = self.rollout()

            t_so_far += np.sum(batch_lens)

            low_V, curr_ll_log_probs = self.evaluate_low_level(batch_obs, batch_acts)
            high_V, curr_hl_log_probs = self.evaluate_high_level(hl_obs, hl_acts)

            lower_level_adv = batch_rew_to_go - low_V.detach()
            higher_level_adv = hl_rew_to_go - high_V.detach()

            # Normalize advantages
            lower_level_adv = (lower_level_adv - lower_level_adv.mean()) / (lower_level_adv.std() + 1e-10)
            higher_level_adv = (higher_level_adv - higher_level_adv.mean()) / (higher_level_adv.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                low_V, curr_ll_log_probs = self.evaluate_low_level(batch_obs, batch_acts)
                high_V, curr_hl_log_probs = self.evaluate_high_level(hl_obs, hl_acts)
                ll_ratio = torch.exp(curr_ll_log_probs - batch_log_probs)
                hl_ratio = torch.exp(curr_hl_log_probs - hl_log_probs)
                surr1_ll = ll_ratio * lower_level_adv
                surr1_hl = hl_ratio * higher_level_adv
                surr2_ll = torch.clamp(ll_ratio, 1 - self.clip, 1 + self.clip) * lower_level_adv
                surr2_hl = torch.clamp(hl_ratio, 1 - self.clip, 1 + self.clip) * higher_level_adv
                ll_loss = (-torch.min(surr1_ll, surr2_ll)).mean()
                hl_loss = (-torch.min(surr1_hl, surr2_hl)).mean()

                self.hl_optim.zero_grad()
                self.ll_optim.zero_grad()
                hl_loss.backward(retain_graph=True)
                ll_loss.backward(retain_graph=True)
                self.hl_optim.step()
                self.ll_optim.step()

                ll_critic_loss = nn.MSELoss()(low_V, batch_rew_to_go)
                self.ll_critic_optim.zero_grad()
                ll_critic_loss.backward()
                self.ll_critic_optim.step()

                hl_critic_loss = nn.MSELoss()(high_V, hl_rew_to_go)
                self.hl_critic_optim.zero_grad()
                hl_critic_loss.backward()
                self.hl_critic_optim.step()

                

    def get_mean_high_level_action(self, obs):
        if self.is_discrete:
            action_logits = self.high_level_policy(obs)
            hl_distribution = Categorical(logits=action_logits)
            hl_action = hl_distribution.sample()
            hl_log_prob = hl_distribution.log_prob(hl_action)
            return hl_action.detach(), hl_log_prob.detach()
        else:
            high_level_mean = self.high_level_policy(obs)
            hl_distribution = MultivariateNormal(high_level_mean, self.covariant_matrix)
            hl_action = hl_distribution.sample()
            hl_log_prob = hl_distribution.log_prob(hl_action)
            return hl_action.detach(), hl_log_prob.detach()

    def get_mean_low_level_action(self, combined_input):
        if self.is_discrete:
            action_logits = self.low_level_policy(combined_input)
            ll_distribution = Categorical(logits=action_logits)
            ll_action = ll_distribution.sample()
            ll_log_prob = ll_distribution.log_prob(ll_action)
            return ll_action.detach(), ll_log_prob.detach()
        else:
            low_level_mean = self.low_level_policy(combined_input)
            ll_distribution = MultivariateNormal(low_level_mean, self.ll_covariant_matrix)
            ll_action = ll_distribution.sample()
            ll_log_prob = ll_distribution.log_prob(ll_action)
            return ll_action.detach(), ll_log_prob.detach()

    def get_high_level_obs(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs)
        if self.is_discrete_obs:
            obs = obs.long()  # Ensure obs is of integer type
            obs = f.one_hot(obs, num_classes=self.obs_dim).float()
        else:
            obs = obs.float()
        return obs

    def evaluate_low_level(self, obs, actions):
        V = self.low_level_critic(obs).squeeze()
        if self.is_discrete:
            action_logits = self.low_level_policy(obs)
            dist = Categorical(logits=action_logits)
            log_prob = dist.log_prob(actions)
        else:
            mean = self.low_level_policy(obs)
            dist = MultivariateNormal(mean, self.ll_covariant_matrix)
            log_prob = dist.log_prob(actions)
        return V, log_prob

    def evaluate_high_level(self, obs, actions):
        V = self.high_level_critic(obs).squeeze()
        if self.is_discrete:
            action_logits = self.high_level_policy(obs)
            dist = Categorical(logits=action_logits)
            log_prob = dist.log_prob(actions)
        else:
            mean = self.high_level_policy(obs)
            dist = MultivariateNormal(mean, self.covariant_matrix)
            log_prob = dist.log_prob(actions)
        return V, log_prob

    def combine_obs_subgoal(self, obs, subgoal):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs)
        if self.is_discrete_obs:
            obs = obs.long()  # Ensure obs is of integer type
            obs = f.one_hot(obs, num_classes=self.obs_dim).float()
        else:
            obs = obs.float()
        if not isinstance(subgoal, torch.Tensor):
            subgoal = torch.tensor(subgoal)
        if self.is_discrete:
            subgoal = subgoal.long()  # Ensure subgoal is of integer type
            subgoal_one_hot = f.one_hot(subgoal, num_classes=self.action_dim).float()
            combined_input = torch.cat([obs, subgoal_one_hot], dim=-1)
        else:
            combined_input = torch.cat([obs, subgoal.float()], dim=-1)
        return combined_input

