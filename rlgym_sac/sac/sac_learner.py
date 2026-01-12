import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

# CleanRL imports
# Assuming rlgym_sac is installed or in path
# We need to make sure we can import these. 
# Since we are in rlgym_sac.sac.sac_learner, cleanrl is in rlgym_sac.cleanrl
from ..cleanrl.sac_continuous_action import Actor, SoftQNetwork
from ..cleanrl.buffers import ReplayBuffer

class PolicyWrapper:
    def __init__(self, actor, device):
        self.actor = actor
        self.device = device
    
    def get_action(self, obs):
        with torch.no_grad():
             if isinstance(obs, np.ndarray):
                 obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
             elif isinstance(obs, torch.Tensor) and obs.device != self.device:
                 obs = obs.to(self.device)
             
             action, log_prob, _ = self.actor.get_action(obs)
             # Return cpu tensors for consumption by downstream numpy casting
             # BatchedAgentManager expects (actions, log_probs)
             return action.cpu(), log_prob.cpu()

class SACLearner:
    def __init__(self,
                 obs_space_size,
                 act_space_size,
                 device="auto",
                 batch_size=256,
                 ent_coef="auto",
                 learning_rate=3e-4,
                 buffer_size=1_000_000,
                 learning_starts=100,
                 train_freq=1,
                 gradient_steps=1,
                 tau=0.005,
                 gamma=0.99,
                 policy_layer_sizes=(256, 256),
                 critic_layer_sizes=(256, 256)):
        
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        
        # Define spaces for CleanRL classes
        # cleanrl buffers expects gymnasium spaces
        self.observation_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(obs_space_size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_space_size,), dtype=np.float32)
        
        # Mock env used by CleanRL classes to get dims
        class MockEnv:
            def __init__(self, obs_space, act_space):
                self.single_observation_space = obs_space
                self.single_action_space = act_space
        
        self.mock_env = MockEnv(self.observation_space, self.action_space)
        
        # Initialization
        self.actor = Actor(self.mock_env, layer_sizes=policy_layer_sizes).to(self.device)
        self.qf1 = SoftQNetwork(self.mock_env, layer_sizes=critic_layer_sizes).to(self.device)
        self.qf2 = SoftQNetwork(self.mock_env, layer_sizes=critic_layer_sizes).to(self.device)
        self.qf1_target = SoftQNetwork(self.mock_env, layer_sizes=critic_layer_sizes).to(self.device)
        self.qf2_target = SoftQNetwork(self.mock_env, layer_sizes=critic_layer_sizes).to(self.device)
        
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=learning_rate)
        
        # Entropy tuning
        self.autotune = (ent_coef == 'auto')
        # Roughly -action_dim
        self.target_entropy = -float(act_space_size) 
        self.alpha = float(ent_coef) if not self.autotune else 1.0 
        
        if self.autotune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.log_alpha = None
            self.a_optimizer = None

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=1, 
        )
        
        self.cumulative_model_updates = 0
        self.policy = PolicyWrapper(self.actor, self.device)

    def add_experience(self, experience):
        states, actions, log_probs, rewards, next_states, dones, truncated = experience
        
        # Efficiently add batch of experience to buffer.
        # rlgym-ppo experience is (N, ...)
        # ReplayBuffer.add expects (n_envs, ...) and numpy inputs
        
        # Convert to numpy if they are tensors (which they likely are from BatchedManager)
        def to_cpu_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return x
            
        states = to_cpu_numpy(states)
        actions = to_cpu_numpy(actions)
        rewards = to_cpu_numpy(rewards)
        next_states = to_cpu_numpy(next_states)
        dones = to_cpu_numpy(dones)
        truncated = to_cpu_numpy(truncated)

        N = len(states)
        
        # NOTE: This loop might be slow for large batches (e.g. 50k steps).
        # ideally we should modify ReplayBuffer to accept batches.
        # But buffers.py is from CleanRL and assumes adding 1 step from N envs.
        # Here we have N steps from effectively 1 flattened stream or many streams (batched agent).
        # We treat it as sequential additions.
        
        for i in range(N):
            info = {}
            if truncated[i]:
                info["TimeLimit.truncated"] = True
            
            # ReplayBuffer expects inputs to be (n_envs, ...)
            # We set n_envs=1 in buffer init, so we expect (1, ...)
            
            self.replay_buffer.add(
                states[i][None, ...],      
                next_states[i][None, ...], 
                actions[i][None, ...],     
                rewards[i].reshape(1),     
                dones[i].reshape(1),       
                [info]
            )

    def learn(self, steps_collected):
        if self.replay_buffer.size() < self.learning_starts:
            return {}
        
        qf_losses = []
        actor_losses = []
        alpha_losses = []
        q_vals = []
        
        # Calculate how many updates to perform
        # Total updates = (steps_collected / train_freq) * gradient_steps
        if self.train_freq > 0:
            updates_to_perform = int(steps_collected / self.train_freq) * self.gradient_steps
        else:
            updates_to_perform = self.gradient_steps # Fallback if train_freq is 0/invalid

        for _ in range(updates_to_perform):
            data = self.replay_buffer.sample(self.batch_size)
            
            # --- Update Q functions ---
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
                qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
            qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()
            
            qf_losses.append(qf_loss.item())
            q_vals.append(qf1_a_values.mean().item())

            # --- Update Actor ---
            # TODO: Add policy delay support if needed (CleanRL has it, we can expose it)
            # For now updating every step
            
            pi, log_pi, _ = self.actor.get_action(data.observations)
            qf1_pi = self.qf1(data.observations, pi)
            qf2_pi = self.qf2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            actor_losses.append(actor_loss.item())

            # --- Update Alpha ---
            if self.autotune:
                with torch.no_grad():
                    _, log_pi, _ = self.actor.get_action(data.observations)
                alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
                alpha_losses.append(alpha_loss.item())

            # --- Update Target Networks ---
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            self.cumulative_model_updates += 1
            
        return {
            "Value Function Loss": np.mean(qf_losses),
            "Policy Loss": np.mean(actor_losses),
            "Alpha": self.alpha,
            "Alpha Loss": np.mean(alpha_losses) if alpha_losses else 0.0,
            "Mean Q Value": np.mean(q_vals)
        }

    def save_to(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(folder_path, "actor.pt"))
        torch.save(self.qf1.state_dict(), os.path.join(folder_path, "qf1.pt"))
        torch.save(self.qf2.state_dict(), os.path.join(folder_path, "qf2.pt"))
        if self.autotune:
             torch.save(self.log_alpha, os.path.join(folder_path, "log_alpha.pt"))
    
    def load_from(self, folder_path):
        # We need to handle potential loading errors if files don't exist
        # But learner.py checks existence of folder.
        
        self.actor.load_state_dict(torch.load(os.path.join(folder_path, "actor.pt"), map_location=self.device))
        self.qf1.load_state_dict(torch.load(os.path.join(folder_path, "qf1.pt"), map_location=self.device))
        self.qf2.load_state_dict(torch.load(os.path.join(folder_path, "qf2.pt"), map_location=self.device))
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        if self.autotune and os.path.exists(os.path.join(folder_path, "log_alpha.pt")):
            self.log_alpha = torch.load(os.path.join(folder_path, "log_alpha.pt"), map_location=self.device)
            # Recreate optimizer if loaded
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
