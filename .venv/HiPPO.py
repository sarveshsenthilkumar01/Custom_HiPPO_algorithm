import torch
import torch as th
from torch import nn
import torch.nn.functional as f
import numpy as np
from hrl_nn_mlp import high_level_policy
from hrl_nn_mlp import low_level_policy
from hrl_nn_mlp import FFNN
from torch.distributions import MultivariateNormal

class HiPPO:
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.llp_obs = llp_obs
        self.llp_actions = llp_actions
        self.high_level_policy = high_level_policy(self.obs_dim, self.action_dim)
        self.low_level_policy = low_level_policy(self.obs_dim, self.action_dim)
        self.low_level_critic = FFNN(self.obs_dim, 1)
        self.high_level_critic = FFNN(self.obs_dim, 1)
        self._init_hyperparameters()
        # this is the variable for the covariant matrix
        self.covariant_var = torch.full(size=(self.action_dim), fill_value = 0.5)
        # we define the actual covariant matrix here
        self.covariant_matrix = torch.diag(self.covariant_var)



    def _init_hyperparameters(self):
        self.timesteps_per_batch = 6400
        self.max_timesteps_per_low_level_episode = 200
        self.max_timesteps_per_high_level_episode = 1600
        self.low_level_steps_per_high_level_steps
        self.gamma = 0.95 # this is up for change too, we need to determine a good lambda value

    def rollout(self):
        batch_obs = [] # batch observations
        batch_acts = [] # batch actions
        batch_log_probs = [] # the logarithmic probabilities of each action
        batch_rewards = [] # batch rewards

        hl_obs = []
        hl_acts = []
        hl_log_probs = []
        hl_rewards = []
        hl_batch_lens
        batch_lens = []  # episodic lengths in batch
        t = 0
        while t < self.timesteps_per_batch:
            # getting the observation from state as a high level observation here
            obs = self.get_high_level_obs(self.env.reset())
            done = False
            ep_rewards = []
            hl_ep_rewards = []
            # we define two separate arrays, one for low level ep rewards,
            # one for high level ep rewards

            ep_t = 0

            while not done and ep_t < self.max_timesteps_per_high_level_episode:
                # we are copying the observation from state to higher level obs
                hl_obs = obs.copy()
                # we are calculating the hihger level action and higher level
                # log probabilities here
                hl_actions, hl_log_probabilities = self.get_mean_high_level_action(obs)
                hl_acts.append(hl_actions)
                hl_log_probs.append(hl_log_probabilities)


                low_level_rewards = []
                low_level_timestep = 0

                for _ in range(self.low_level_steps_per_high_level_steps):
                    # here, we are combining the observations with the higher level policy's actions
                    # to feed as input to the lower level policy
                    # essentially creating our own subgoals
                    combined_obs = self.combine_high_level_obs(obs, hl_actions)
                    ll_actions, ll_log_probabilities = self.get_mean_low_level_action(combined_obs)
                    obs, reward, done, _ = self.env.step(ll_actions)

                    batch_obs.append(combined_obs)
                    batch_acts.append(ll_actions)
                    batch_log_probs.append(ll_log_probabilities)
                    ep_rewards.append(reward)
                    obs = next_obs
                    t += 1
                    low_level_steps += 1
                    low_level_rewards.append(reward)

                    if done or t >= self.timesteps_per_batch:
                        break

                    # here we are calculating the high level rewards as a sum of the
                    # low level rewards, and appending it to the list
                hl_reward = sum(low_level_rewards)
                hl_ep_rewards.append(hl_reward)

                ep_t += low_level_steps

                if done or t >= self.timesteps_per_batch:
                    break

            # here we append the total rewards of the batch to the lists below
            batch_lens.append(ep_t)
            batch_rewards.append(ep_rewards)
            hl_batch_lens.append(len(hl_ep_rewards))
            hl_batch_rewards.append(hl_ep_rewards)


        batch_rtgs = [] # batch rewards to go
        hl_rtgs = []
        batch_obs = torch.tensor(batch_obs, dtype = torch.float)
        batch_acts = torch.tensor(batch_acts, dtype = torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype = torch.float)
        batch_rew_to_go = self.compute_rtgs(batch_rews)
        hl_obs = torch.tensor(hl_obs, dtype = torch.float)
        hl_acts = torch.tensor(hl_acts, dtype = torch.float)
        hl_log_probs = torch.tensor(hl_log_probs, dtype = torch.float)
        hl_rew_to_go = torch.tensor(hl_rew_to_go, dtype = torch.float)
        return batch_obs, batch_acts, batch_log_probs, batch_rew_to_go, batch_lens, hl_obs, hl_acts, hl_log_probs, hl_rew_to_go

    def compute_rtgs(self, batch_rews):
        # the shape will be the number of timesteps per episode
        batch_rtgs = []
        # we need to iterate through every episode backwards in order to maintain the same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype = torch.float)
        return batch_rtgs
    # need to check if you should implement a rewards system here that has high level batch rewards and low level batch rewards within
    # the learn function
    def learn(self, total_timesteps):
        t_so_far = 0
        while(t_so_far < total_timesteps):
            # episode rewards
            batch_obs, batch_acts, batch_log_probs, batch_rew_to_go, batch_lens,
            hl_obs, hl_acts, hl_log_probs, hl_rew_to_go, hl_lens = self.rollout()

            lower_level_V = self.evaluate_low_level(batch_obs + hl_acts)
            higher_level_V = self.evaluate_high_level(hl_obs)

            lower_level_adv = batch_rew_to_go - lower_level_V.detach()
            higher_level_adv = hl_rew_to_go - higher_level_V.detach()

            # normalizing the advantage for lower and higher level policy
            lower_level_adv_norm = (lower_level_adv - lower_level_adv.mean()) / (lower_level_adv.std() + 1e-10) # you add 1e-10 just to make sure there is no division by 0
            higher_level_adv_norm = (higher_level_adv - higher_level_adv.mean()) / (higher_level_adv.std() + 1e-10)




    # this function is to calculate the standard deviation for filling the covariance matrix
    def standard_dev_cal(self, action):
        mean_action = self.high_level_policy(action)
        difference = action - mean_action
        standard_dev = torch.exp(log_prob / (-0.5 * diff_shape[0]))
        return standard_dev

    def get_mean_high_level_action(self, obs):
        high_level_mean = self.high_level_policy(obs)
        hl_distribution = MultivariateNormal(high_level_mean, self.covariant_matrix)
        hl_action = hl_distribution.sample()
        hl_log_prob = hl_distribution.log_prob(action)
        return hl_action.detach().numpy(), hl_log_prob.detach().numpy()

    def get_mean_low_level_action(self, combined_input):
        low_level_mean = self.low_level_policy(combined_input)
        self.ll_covariance_matrix = torch.full(size = self.action_dim, fill_value=standard_dev_cal(combined_input))
        low_level_distribution = MultivariateNormal(low_level_mean, self.ll_covariant_matrix)
        ll_action = low_level_distribution.sample()
        ll_log_prob = low_level_distribution.log_prob(ll_action)
        return ll_action.detach().numpy(), ll_log_prob.detach().numpy()

    # to compute low_level_observations
    def get_low_level_obs(self, obs, high_level_action):
        return torch.cat((torch.tensor(obs), high_level_action), dim=-1)

    # to compute high_level_observation
    def get_high_level_obs(self, obs):
        return torch.tensor(obs, dtype = torch.float)

    # this is to calculate the predicted values
    def evaluate_low_level(self, obs):
        V = self.low_level_critic(obs).squeeze()
        return V

    def evaluate_high_level(self, obs):
        V = self.high_level_critic(obs).squeeze()
        return V

    def combine_obs_subgoal(self, obs, subgoal):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype = torch.float)
        if not isinstance(subgoal, torch.Tensor):
            subgoal = torch.tensor(subgoal, dtype = torch.float)
        combined_input = torch.cat([obs, subgoal], dim=-1)
        return combined_input
