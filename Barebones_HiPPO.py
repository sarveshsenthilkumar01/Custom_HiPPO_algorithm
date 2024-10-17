import torch
import numpy as np
import torch.nn.functional as F
import gymnasium
from gymnasium.wrappers import TimeLimit
from torch.distributions import MultivariateNormal, Categorical
from Neural_network import high_level_policy, low_level_policy, FFNN_hlp, FFNN_llp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython import display
import torch.nn as nn

class HiPPO:
    def __init__(self, env):
        self.env = env
        self.isContinuous = False
        self.obs_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gymnasium.spaces.Box):
            self.act_dim = env.action_space.shape[0]
            self.isContinuous = True
        elif isinstance(env.action_space, gymnasium.spaces.Discrete):
            self.act_dim = env.action_space.n
            self.isContinuous = False

        # Define the device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # Confirmation

        # Initialize high-level and low-level policies
        self.hlp = high_level_policy(self.obs_dim, self.act_dim).to(self.device)
        self.llp = low_level_policy(self.obs_dim + self.act_dim, self.act_dim).to(self.device)

        # Initialize critic networks for high-level and low-level policies
        self.hl_critic = FFNN_hlp(self.obs_dim, 1).to(self.device)
        self.ll_critic = FFNN_llp(self.obs_dim + self.act_dim, 1).to(self.device)

        # Initialize weights
        self.initialize_weights()

        # Set hyperparameters
        self.hyperparameters()

        # Initialize covariance matrix for action distributions
        self.covariance_var = torch.full(size=(self.act_dim,), fill_value=0.5).to(self.device)
        self.covariance_matrix = torch.diag(self.covariance_var).to(self.device)

        # Initialize optimizers for policies and critics
        self.llp_policy_optimizer = torch.optim.Adam(self.llp.parameters(), lr=self.ll_lr)
        self.hlp_policy_optimizer = torch.optim.Adam(self.hlp.parameters(), lr=self.hl_lr)
        self.llp_critic_optimizer = torch.optim.Adam(self.ll_critic.parameters(), lr=self.ll_lr)
        self.hlp_critic_optimizer = torch.optim.Adam(self.hl_critic.parameters(), lr=self.hl_lr)

        # Variables to store rewards per episode for plotting
        self.rewards_per_episode = []
        # Initialize live plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')
        self.ax.set_title('Rewards per Episode')

    def initialize_weights(self):
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                # Initialize with small weights
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                torch.nn.init.zeros_(m.bias)

        self.hlp.apply(init_weights)
        self.llp.apply(init_weights)
        self.hl_critic.apply(init_weights)
        self.ll_critic.apply(init_weights)

    def hyperparameters(self):
        self.timesteps_per_batch = 5000  # Adjusted for more frequent updates
        self.max_timesteps_per_episode = 1000  # Adjusted to desired maximum
        self.hlp_timesteps = 15
        self.gamma = 0.99  # Discount factor
        self.lam = 0.95    # GAE lambda parameter
        self.epoch_number = 10  # Number of epochs per update
        self.clip = 0.2
        self.hl_lr = 3e-4  # High-level policy learning rate
        self.ll_lr = 1e-3  # Low-level policy learning rate
        self.minibatch_number = 20
        self.llp_ent_coeff = 0.01
        self.hlp_ent_coeff = 0.01
        self.kl_threshold = 0.01
        self.ll_target_kl = 0.03
        self.hl_target_kl = 0.05
        self.min_lr = 1e-5  # Minimum learning rate for safety
        self.max_lr = 1e-2  # Maximum learning rate for safety

    def rollout(self):
        # Initialize storage for rollout data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []        # Collect rewards per timestep
        batch_dones = []          # Collect dones per timestep
        batch_lengths = []

        hlp_obs = []
        hlp_acts = []
        hlp_log_probs = []
        hlp_rewards = []          # Collect high-level rewards per decision point
        hlp_dones = []
        t = 0

        with torch.no_grad():  # Prevent gradient tracking during rollout
            while t < self.timesteps_per_batch:
                hlp_episode_rewards = 0
                episode_rewards = []
                obs, info = self.env.reset()
                done = False
                hl_done = False
                for ep_t in range(self.max_timesteps_per_episode):
                    t += 1

                    # At the start or every hlp_timesteps, sample a new high-level action
                    if ep_t % self.hlp_timesteps == 0:
                        # Append accumulated hlp_episode_rewards if not the first time
                        if ep_t > 0:
                            hlp_rewards.append(hlp_episode_rewards)
                            hlp_dones.append(float(hl_done))
                            hlp_episode_rewards = 0

                        # Get new high-level action based on the latest observation
                        hlp_obs.append(torch.from_numpy(obs).float().to(self.device))
                        hl_action, hl_log_prob = self.get_action_hlp(obs)
                        hl_action = np.clip(hl_action, self.env.action_space.low, self.env.action_space.high)
                        hl_action_tensor = torch.from_numpy(hl_action).float().to(self.device)
                        hlp_acts.append(hl_action_tensor)
                        hlp_log_probs.append(hl_log_prob)
                        current_hl_action = hl_action
                        _, _, terminated1, truncated1, _ = self.env.step(hl_action)
                        hl_done = terminated1 or truncated1

                    # Low-level policy action
                    llp_obs = self.create_subgoal(obs, current_hl_action).to(self.device)
                    batch_obs.append(llp_obs)
                    ll_action, ll_log_prob = self.get_action_llp(obs, current_hl_action)
                    ll_action = np.clip(ll_action, self.env.action_space.low, self.env.action_space.high)

                    # Execute low-level action
                    obs, reward, terminated, truncated, info = self.env.step(ll_action)
                    done = terminated or truncated

                    # Collect low-level data
                    batch_rewards.append(reward)    # Collect reward per timestep
                    batch_acts.append(torch.from_numpy(ll_action).float().to(self.device))
                    batch_log_probs.append(ll_log_prob)
                    batch_dones.append(float(done))  # Convert to float for tensor operations

                    # Accumulate high-level rewards
                    hlp_episode_rewards += reward
                    episode_rewards.append(reward)

                    if done:
                        # Append the last accumulated high-level reward
                        hlp_rewards.append(hlp_episode_rewards)
                        hlp_dones.append(float(done))
                        hlp_episode_rewards = 0

                        # Collecting the length of the episode
                        batch_lengths.append(ep_t + 1)
                        self.rewards_per_episode.append(sum(episode_rewards))
                        if terminated:
                            print(f"Episode terminated due to failure at timestep: {ep_t + 1}, Total Reward: {sum(episode_rewards)}")
                        elif truncated:
                            print(f"Episode reached max timesteps at timestep: {ep_t + 1}, Total Reward: {sum(episode_rewards)}")
                        else:
                            print(f"Episode ended at timestep: {ep_t + 1}, Total Reward: {sum(episode_rewards)}")
                        self.update_plot()  # Update the plot after each episode
                        break

                else:
                    # If max_timesteps_per_episode reached without done
                    if hlp_episode_rewards > 0:
                        hlp_rewards.append(hlp_episode_rewards)
                        hlp_dones.append(float(done))
                    batch_lengths.append(self.max_timesteps_per_episode)
                    self.rewards_per_episode.append(sum(episode_rewards))
                    print(f"Episode reached max timesteps: {self.max_timesteps_per_episode}, Total Reward: {sum(episode_rewards)}")
                    self.update_plot()

        # After all rollouts, stack the tensors
        try:
            batch_obs = torch.stack(batch_obs).to(self.device)
            batch_acts = torch.stack(batch_acts).to(self.device)
            batch_log_probs = torch.stack(batch_log_probs).to(self.device)
            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float).to(self.device)
            batch_dones = torch.tensor(batch_dones, dtype=torch.float).to(self.device)

            hlp_obs = torch.stack(hlp_obs).to(self.device)
            hlp_acts = torch.stack(hlp_acts).to(self.device)
            hlp_log_probs = torch.stack(hlp_log_probs).to(self.device)
            hlp_rewards = torch.tensor(hlp_rewards, dtype=torch.float).to(self.device)
            hlp_dones = torch.tensor(hlp_dones, dtype=torch.float).to(self.device)

            # Debugging Statements
            print(f"batch_obs shape: {batch_obs.shape}")
            print(f"hlp_obs shape: {hlp_obs.shape}")
            print(f"hlp_rewards shape: {hlp_rewards.shape}")
        except RuntimeError as e:
            print("Error during torch.stack:")
            print(e)
            raise e

        # Return all collected data
        return (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_rewards,        # Return rewards per timestep
            batch_lengths,
            hlp_obs,
            hlp_acts,
            hlp_log_probs,
            hlp_rewards,
            hlp_dones,
            batch_dones
        )

    def learn(self, total_timesteps):
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging
        t_so_far = 0
        episode_count = 0
        while t_so_far < total_timesteps:
            try:
                (
                    batch_obs,
                    batch_acts,
                    batch_log_probs,
                    batch_rewards,
                    batch_lens,
                    hlp_obs,
                    hlp_acts,
                    hlp_log_probs,
                    hlp_rewards,
                    hlp_dones,
                    batch_dones
                ) = self.rollout()

            except ValueError as ve:
                print(f"Rollout Error: {ve}")
                continue  # Skip this batch and continue training
            except Exception as e:
                print(f"Unexpected Error during rollout: {e}")
                continue  # Skip this batch and continue training

            episode_count += len(batch_lens)
            print(f"Total Episodes so far: {episode_count}")

            # Verify consistency
            assert hlp_obs.shape[0] == hlp_rewards.shape[0], (
                f"Mismatch between hlp_obs ({hlp_obs.shape[0]}) and hlp_rewards ({hlp_rewards.shape[0]}). "
                "High-level observations and rewards sizes do not match."
            )

            # Calculate value estimates for low-level and high-level policies
            ll_V = self.ll_critic(batch_obs).squeeze()
            hl_V = self.hl_critic(hlp_obs).squeeze()

            # Compute GAE advantages
            ll_adv = self.compute_gae(batch_rewards, ll_V.detach(), batch_dones)
            hl_adv = self.compute_gae(hlp_rewards, hl_V.detach(), hlp_dones)

            # Normalize advantages
            ll_adv = (ll_adv - ll_adv.mean()) / (ll_adv.std() + 1e-8)
            hl_adv = (hl_adv - hl_adv.mean()) / (hl_adv.std() + 1e-8)

            # Convert actions and log_probs to tensors if not already
            #batch_log_probs = torch.stack(batch_log_probs).to(self.device)
            #hlp_log_probs = torch.stack(hlp_log_probs).to(self.device)

            steps = batch_obs.size(0)
            inds = np.arange(steps)

            # Set your desired minibatch size
            minibatch_size = 64  # For example, adjust based on your preference

            # Learning rate scheduling
            ll_frac = (t_so_far - 1.0) / total_timesteps
            hl_frac = (t_so_far - 1.0) / total_timesteps

            new_hl_lr = self.hl_lr * (1.0 - hl_frac)
            new_ll_lr = self.ll_lr * (1.0 - ll_frac)

            # Update learning rates
            for param_group in self.hlp_policy_optimizer.param_groups:
                param_group['lr'] = new_hl_lr
            for param_group in self.hlp_critic_optimizer.param_groups:
                param_group['lr'] = new_hl_lr

            for param_group in self.llp_policy_optimizer.param_groups:
                param_group['lr'] = new_ll_lr
            for param_group in self.llp_critic_optimizer.param_groups:
                param_group['lr'] = new_ll_lr

            # Training loop over epochs
            for _ in range(self.epoch_number):
                # Shuffle indices for mini-batch updates
                np.random.shuffle(inds)

                # Calculate the number of mini-batches
                num_minibatches = max(1, steps // minibatch_size)

                for i in range(num_minibatches):
                    start = i * minibatch_size
                    end = start + minibatch_size
                    llp_idx = inds[start:end]

                    # For high-level policy, adjust indices accordingly
                    hlp_steps = hlp_obs.size(0)
                    hlp_inds = np.arange(hlp_steps)
                    np.random.shuffle(hlp_inds)
                    hlp_minibatch_size = max(1, hlp_steps // num_minibatches)
                    hlp_start = i * hlp_minibatch_size
                    hlp_end = hlp_start + hlp_minibatch_size
                    hlp_idx = hlp_inds[hlp_start:hlp_end]

                    # Extract mini-batch data for low-level policy
                    mini_llp_obs = batch_obs[llp_idx]
                    mini_llp_acts = batch_acts[llp_idx]
                    mini_llp_log_probs = batch_log_probs[llp_idx]
                    mini_llp_advantage = ll_adv[llp_idx]
                    mini_llp_rtgs = ll_V[llp_idx] + mini_llp_advantage  # Targets for value function

                    # Extract mini-batch data for high-level policy
                    mini_hlp_obs = hlp_obs[hlp_idx]
                    mini_hlp_acts = hlp_acts[hlp_idx]
                    mini_hlp_log_probs = hlp_log_probs[hlp_idx]
                    mini_hlp_advantage = hl_adv[hlp_idx]
                    mini_hlp_rtgs = hl_V[hlp_idx] + mini_hlp_advantage  # Targets for value function

                    # Evaluate policies on mini-batches
                    ll_V_mb, curr_ll_probs, llp_entropy_mb, hl_V_mb, curr_hl_probs, hlp_entropy_mb = self.evaluate(
                        mini_llp_obs, mini_llp_acts, mini_hlp_obs, mini_hlp_acts
                    )

                    # Calculate ratios and surrogate losses for low-level policy
                    ll_ratio = torch.exp(curr_ll_probs - mini_llp_log_probs)
                    ll_surr1 = ll_ratio * mini_llp_advantage
                    ll_surr2 = torch.clamp(ll_ratio, 1 - self.clip, 1 + self.clip) * mini_llp_advantage
                    llp_loss = (-torch.min(ll_surr1, ll_surr2)).mean()

                    # Add entropy bonus
                    llp_loss = llp_loss - self.llp_ent_coeff * llp_entropy_mb.mean()

                    # Update low-level policy
                    self.llp_policy_optimizer.zero_grad()
                    llp_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.llp.parameters(), max_norm=0.5)
                    self.llp_policy_optimizer.step()

                    # Calculate ratios and surrogate losses for high-level policy
                    hl_ratio = torch.exp(curr_hl_probs - mini_hlp_log_probs)
                    hl_surr1 = hl_ratio * mini_hlp_advantage
                    hl_surr2 = torch.clamp(hl_ratio, 1 - self.clip, 1 + self.clip) * mini_hlp_advantage
                    hlp_loss = (-torch.min(hl_surr1, hl_surr2)).mean()

                    # Add entropy bonus
                    hlp_loss = hlp_loss - self.hlp_ent_coeff * hlp_entropy_mb.mean()

                    # Update high-level policy
                    self.hlp_policy_optimizer.zero_grad()
                    hlp_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.hlp.parameters(), max_norm=0.5)
                    self.hlp_policy_optimizer.step()

                    # Update low-level value function
                    ll_V_pred = self.ll_critic(mini_llp_obs).squeeze()
                    ll_value_loss = F.mse_loss(ll_V_pred, mini_llp_rtgs.detach())

                    self.llp_critic_optimizer.zero_grad()
                    ll_value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ll_critic.parameters(), max_norm=0.5)
                    self.llp_critic_optimizer.step()

                    # Update high-level value function
                    hl_V_pred = self.hl_critic(mini_hlp_obs).squeeze()
                    hl_value_loss = F.mse_loss(hl_V_pred, mini_hlp_rtgs.detach())

                    self.hlp_critic_optimizer.zero_grad()
                    hl_value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.hl_critic.parameters(), max_norm=0.5)
                    self.hlp_critic_optimizer.step()

                    # Calculate KL divergence
                    ll_kl_div = (mini_llp_log_probs - curr_ll_probs).mean()
                    hl_kl_div = (mini_hlp_log_probs - curr_hl_probs).mean()

                    # Optional: Log losses and KL divergence
                    print(
                        f"LLP Loss: {llp_loss.item():.4f}, HLP Loss: {hlp_loss.item():.4f}, "
                        f"LLP Value Loss: {ll_value_loss.item():.4f}, HLP Value Loss: {hl_value_loss.item():.4f}, "
                        f"LLP KL Div: {ll_kl_div.item():.4f}, HLP KL Div: {hl_kl_div.item():.4f}"
                    )

                    # Early stopping based on KL divergence
                    if ll_kl_div > self.kl_threshold:
                        print(f"LLP KL divergence too high ({ll_kl_div.item():.4f}), stopping updates for this epoch.")
                        break  # Break out of the mini-batch loop or epoch loop

                    if hl_kl_div > self.kl_threshold:
                        print(f"HLP KL divergence too high ({hl_kl_div.item():.4f}), stopping updates for this epoch.")
                        break

                    # Adjust learning rates based on KL divergence
                    # Check KL divergence for low-level policy
                    if ll_kl_div > self.ll_target_kl * 1.5:
                        print(f"Low-level policy KL divergence too high ({ll_kl_div.item():.4f}), reducing learning rate.")
                        for param_group in self.llp_policy_optimizer.param_groups:
                            param_group['lr'] *= 0.5  # Reduce learning rate
                    elif ll_kl_div < self.ll_target_kl / 1.5:
                        print(f"Low-level policy KL divergence too low ({ll_kl_div.item():.4f}), increasing learning rate.")
                        for param_group in self.llp_policy_optimizer.param_groups:
                            param_group['lr'] *= 1.5  # Increase learning rate

                    # Ensure learning rate stays within reasonable bounds
                    for param_group in self.llp_policy_optimizer.param_groups:
                        param_group['lr'] = max(self.min_lr, min(param_group['lr'], self.max_lr))

                    # Check KL divergence for high-level policy
                    if hl_kl_div > self.hl_target_kl * 1.5:
                        print(f"High-level policy KL divergence too high ({hl_kl_div.item():.4f}), reducing learning rate.")
                        for param_group in self.hlp_policy_optimizer.param_groups:
                            param_group['lr'] *= 0.5
                    elif hl_kl_div < self.hl_target_kl / 1.5:
                        print(f"High-level policy KL divergence too low ({hl_kl_div.item():.4f}), increasing learning rate.")
                        for param_group in self.hlp_policy_optimizer.param_groups:
                            param_group['lr'] *= 1.5

                    # Ensure learning rate stays within reasonable bounds
                    for param_group in self.hlp_policy_optimizer.param_groups:
                        param_group['lr'] = max(self.min_lr, min(param_group['lr'], self.max_lr))

                # Update the timestep counter
                t_so_far += np.sum(batch_lens)
                print(f"Total timesteps so far: {t_so_far}/{total_timesteps}")

    def get_action_llp(self, obs, act):
        # Convert action and observation to tensors
        act = torch.from_numpy(act).float().to(self.device)
        obs = torch.from_numpy(obs).float().to(self.device)
        combined_input = torch.cat([obs, act], dim=-1)

        mean = self.llp(combined_input)
        if self.isContinuous:
            ll_dist = MultivariateNormal(mean, self.covariance_matrix)
            action = ll_dist.sample()
            # Apply tanh to bound actions between [-1, 1]
            action = torch.tanh(action)
            # Scale actions to the action space
            action = action * torch.from_numpy(self.env.action_space.high).to(self.device)
        else:
            ll_dist = Categorical(logits=mean)
            action = ll_dist.sample()

        log_prob = ll_dist.log_prob(action)

        # Convert action to CPU and NumPy for environment compatibility
        action_np = action.detach().cpu().numpy()
        action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)

        return action_np, log_prob

    def get_action_hlp(self, obs):
        # Convert observation to tensor
        obs = torch.from_numpy(obs).float().to(self.device)
        mean = self.hlp(obs)
        if self.isContinuous:
            hl_dist = MultivariateNormal(mean, self.covariance_matrix)
            action = hl_dist.sample()
            action = torch.tanh(action)
            action = action * torch.from_numpy(self.env.action_space.high).to(self.device)
        else:
            hl_dist = Categorical(logits=mean)
            action = hl_dist.sample()

        log_prob = hl_dist.log_prob(action)

        # Convert action to CPU and NumPy for environment compatibility
        action_np = action.detach().cpu().numpy()
        action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)

        return action_np, log_prob

    def compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t < len(rewards) - 1:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]
            else:
                next_value = 0
                next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[t] = gae
            if dones[t]:
                gae = 0  # Reset advantage at episode boundaries
        return advantages

    def evaluate(self, ll_obs, ll_acts, hl_obs, hl_acts):
        # Evaluate critic networks
        llp_V = self.ll_critic(ll_obs).squeeze()
        hlp_V = self.hl_critic(hl_obs).squeeze()

        # Evaluate low-level policy
        ll_mean = self.llp(ll_obs)
        if self.isContinuous:
            ll_dist = MultivariateNormal(ll_mean, self.covariance_matrix)
        else:
            ll_dist = Categorical(logits=ll_mean)
        ll_log_probs = ll_dist.log_prob(ll_acts)
        ll_ent = ll_dist.entropy()
        # Evaluate high-level policy
        hl_mean = self.hlp(hl_obs)
        if self.isContinuous:
            hl_dist = MultivariateNormal(hl_mean, self.covariance_matrix)
        else:
            hl_dist = Categorical(logits=hl_mean)
        hl_log_probs = hl_dist.log_prob(hl_acts)
        hl_ent = hl_dist.entropy()
        return llp_V, ll_log_probs, ll_ent, hlp_V, hl_log_probs, hl_ent

    def create_subgoal(self, obs, act):
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        act_tensor = torch.from_numpy(act).float().to(self.device)
        combined_input = torch.cat([obs_tensor, act_tensor], dim=-1)
        return combined_input

    def update_plot(self):
        self.line.set_xdata(range(len(self.rewards_per_episode)))
        self.line.set_ydata(self.rewards_per_episode)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # Pause briefly to allow the plot to update

    def plot_rewards(self):
        plt.figure()
        plt.plot(self.rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards per Episode')
        plt.savefig('rewards_per_episode.png')  # Save the plot to a file
        plt.show()  # Display the plot window

# Initialize the environment and the HiPPO model
if __name__ == "__main__":
    from gymnasium.wrappers import TimeLimit

    # Desired maximum timesteps per episode
    desired_max_steps = 1000  # Adjusted to desired value

    # Initialize the environment without rendering
    env = gymnasium.make("Pendulum-v1", render_mode=None)  # Disable rendering for faster training

    # Apply the TimeLimit wrapper to override max_episode_steps
    env = TimeLimit(env, max_episode_steps=desired_max_steps)

    # Verify the new max_episode_steps
    spec = env.spec
    print("New Environment spec timesteps:", spec.max_episode_steps)

    # Initialize the HiPPO model
    model = HiPPO(env)
    model.learn(1500000)
    plt.show()
