import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
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
                 policy_delay=1,
                 max_updates_per_iter=None,
                 tau=0.005,
                 gamma=0.99,
                 policy_layer_sizes=(256, 256),
                 critic_layer_sizes=(256, 256),
                 use_amp=True):
        
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.policy_delay = max(1, policy_delay)
        self.max_updates_per_iter = max_updates_per_iter if max_updates_per_iter is None else int(max_updates_per_iter)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.autocast_device = "cuda"
        # Gracefully handle older torch versions that lack device_type on GradScaler
        try:
            self.scaler = GradScaler(device_type=self.autocast_device, enabled=self.use_amp)
            self._use_device_type_ctx = True
        except TypeError:
            self.scaler = GradScaler(enabled=self.use_amp)
            self._use_device_type_ctx = False
        
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
        
        # Compile two separate graphs: one for full update, one for critic only
        # We use default compilation (inductor) which offers good speedup without the instability of reduce-overhead/cudagraphs
        try:
             self._update_chunk_compiled = torch.compile(self._update_chunk_internal)
             self._update_all_compiled = torch.compile(self._update_all_internal)
             self._update_critic_compiled = torch.compile(self._update_critic_only_internal)
             print("Model update functions successfully compiled with torch.compile (default).")
        except Exception as e:
             print(f"Failed to compile: {e}. Falling back to eager execution.")
             self._update_chunk_compiled = self._update_chunk_internal
             self._update_all_compiled = self._update_all_internal
             self._update_critic_compiled = self._update_critic_only_internal

    def _autocast_context(self):
        try:
            return autocast(device_type=self.autocast_device, enabled=self.use_amp)
        except TypeError:
            return autocast(enabled=self.use_amp)

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

        # Vectorized addition using add_batch from modified buffers.py
        # We pass truncated array directly as 'infos' to piggyback on the implementation
        
        # CleanRL buffers expect (N, n_envs, ...) but we have (N, ...)
        # add_batch handles reshaping internally assuming n_envs is consistent with buffer init (which is 1)
        
        self.replay_buffer.add_batch(
            states,
            next_states,
            actions,
            rewards,
            dones,
            truncated # Passed as infos
        )

    def _update_critic_logic(self, obs, act, next_obs, rew, done):
        """Shared critic logic to avoid code duplication"""
        with self._autocast_context():
            with torch.no_grad():
                current_alpha = self.log_alpha.exp().detach() if self.autotune else torch.tensor(self.alpha, device=self.device)
                
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_obs)
                qf1_next_target = self.qf1_target(next_obs, next_state_actions)
                qf2_next_target = self.qf2_target(next_obs, next_state_actions)
                
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - current_alpha * next_state_log_pi
                next_q_value = rew.flatten() + (1 - done.flatten()) * self.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = self.qf1(obs, act).view(-1)
            qf2_a_values = self.qf2(obs, act).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

        # set_to_none=False is required for CUDA Graphs (reduce-overhead) to keep static gradient buffers
        self.q_optimizer.zero_grad(set_to_none=False)
        if self.use_amp:
            self.scaler.scale(qf_loss).backward()
            self.scaler.step(self.q_optimizer)
        else:
            qf_loss.backward()
            self.q_optimizer.step()
        
        return qf_loss.detach().clone(), qf1_a_values.detach().mean().clone()

    def _update_critic_only_internal(self, obs, act, next_obs, rew, done):
        qf_loss, q_val_mean = self._update_critic_logic(obs, act, next_obs, rew, done)
        if self.use_amp:
            self.scaler.update()
        return qf_loss.detach().clone(), q_val_mean.detach().clone()

    def _update_all_internal(self, obs, act, next_obs, rew, done):
        # 1. Update Critic
        qf_loss_val, q_val_mean = self._update_critic_logic(obs, act, next_obs, rew, done)

        # 2. Update Actor
        with self._autocast_context():
            pi, log_pi, _ = self.actor.get_action(obs)
            qf1_pi = self.qf1(obs, pi)
            qf2_pi = self.qf2(obs, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            
            # Re-fetch alpha (graph safe)
            current_alpha = self.log_alpha.exp() if self.autotune else torch.tensor(self.alpha, device=self.device)
            if not self.autotune:
                 current_alpha = current_alpha.detach() # Explicit detach for fixed alpha

            actor_loss = ((current_alpha * log_pi) - min_qf_pi).mean()

        # set_to_none=False for CUDA Graphs
        self.actor_optimizer.zero_grad(set_to_none=False)
        if self.use_amp:
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)
        else:
            actor_loss.backward()
            self.actor_optimizer.step()
        
        actor_loss_val = actor_loss.detach().clone()

        # 3. Update Alpha
        alpha_val = torch.tensor(0.0, device=self.device)
        alpha_loss_val = torch.tensor(0.0, device=self.device)

        if self.autotune:
            with self._autocast_context():
                with torch.no_grad():
                    _, log_pi_alpha, _ = self.actor.get_action(obs)
                alpha_loss = (-self.log_alpha.exp() * (log_pi_alpha + self.target_entropy)).mean()

            # set_to_none=False for CUDA Graphs
            self.a_optimizer.zero_grad(set_to_none=False)
            if self.use_amp:
                self.scaler.scale(alpha_loss).backward()
                self.scaler.step(self.a_optimizer)
            else:
                alpha_loss.backward()
                self.a_optimizer.step()
            
            alpha_loss_val = alpha_loss.detach().clone()
            alpha_val = self.log_alpha.exp().detach().clone()
        else:
            alpha_val = torch.tensor(self.alpha, device=self.device)

        if self.use_amp:
            self.scaler.update()

        # 4. Update Targets
        with torch.no_grad():
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.lerp_(param.data, self.tau)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.lerp_(param.data, self.tau)

        return qf_loss_val, actor_loss_val, alpha_loss_val, q_val_mean, alpha_val

    def _update_chunk_internal(self, obs_chunk, act_chunk, next_obs_chunk, rew_chunk, done_chunk, chunk_size):
        qf_loss_sum = torch.tensor(0.0, device=self.device)
        actor_loss_sum = torch.tensor(0.0, device=self.device)
        alpha_loss_sum = torch.tensor(0.0, device=self.device)
        q_val_sum = torch.tensor(0.0, device=self.device)
        
        for i in range(chunk_size):
            obs, act, next_obs, rew, done = obs_chunk[i], act_chunk[i], next_obs_chunk[i], rew_chunk[i], done_chunk[i]
            
            # Assume strict alignment of chunk updates to policy_delay
            # Caller must ensure chunk_size is multiple of policy_delay and start_idx is aligned
            if i % self.policy_delay == 0:
                qf, act_l, alph_l, qv, _ = self._update_all_internal(obs, act, next_obs, rew, done)
                qf_loss_sum += qf
                actor_loss_sum += act_l
                alpha_loss_sum += alph_l
                q_val_sum += qv
            else:
                qf, qv = self._update_critic_only_internal(obs, act, next_obs, rew, done)
                qf_loss_sum += qf
                q_val_sum += qv
                
        return qf_loss_sum, actor_loss_sum, alpha_loss_sum, q_val_sum

    def learn(self, steps_collected):
        if self.replay_buffer.size() < self.learning_starts:
            return {}
        
        # Accumulators on GPU
        qf_loss_sum = torch.tensor(0.0, device=self.device)
        actor_loss_sum = torch.tensor(0.0, device=self.device)
        alpha_loss_sum = torch.tensor(0.0, device=self.device)
        q_val_sum = torch.tensor(0.0, device=self.device)
        actor_updates_count = torch.tensor(0.0, device=self.device)
        
        # Timers
        update_time = 0.0
        sample_time = 0.0
        reshape_time = 0.0
        step_time = 0.0
        
        # Calculate how many updates to perform
        # Total updates = (steps_collected / train_freq) * gradient_steps
        if self.train_freq > 0:
            updates_to_perform = int(steps_collected / self.train_freq) * self.gradient_steps
        else:
            updates_to_perform = self.gradient_steps # Fallback if train_freq is 0/invalid

        if self.max_updates_per_iter is not None:
            updates_to_perform = min(updates_to_perform, self.max_updates_per_iter)

        # Optimization: Sample everything at once (or in large chunks)
        # We chunk into small blocks to utilize the compiled chunk function which unrolls the loop
        
        CHUNK_SIZE = 120 # Higher chunk size (multiple of policy_delay=3) to amortize kernel launches
        update_idx = 0
        
        t0_total = time.perf_counter()

        while update_idx < updates_to_perform:
            current_chunk = min(CHUNK_SIZE, updates_to_perform - update_idx)
            
            t0_sample = time.perf_counter()
            # Sample
            data = self.replay_buffer.sample(self.batch_size * current_chunk)
            t1_sample = time.perf_counter()
            sample_time += (t1_sample - t0_sample)
            
            # Reshape data to (Chunk, Batch, Dim)
            def reshape_tensor(x):
                new_shape = (current_chunk, self.batch_size) + x.shape[1:]
                return x.view(new_shape)

            obs_chunk = reshape_tensor(data.observations)
            act_chunk = reshape_tensor(data.actions)
            next_obs_chunk = reshape_tensor(data.next_observations)
            rew_chunk = reshape_tensor(data.rewards)
            done_chunk = reshape_tensor(data.dones)
            t2_reshape = time.perf_counter()
            reshape_time += (t2_reshape - t1_sample)
            
            if current_chunk == CHUNK_SIZE and update_idx % self.policy_delay == 0:
                # Use the compiled chunk update
                # Requirement: chunk_size MUST be multiple of policy_delay to maintain alignment
                # CHUNK_SIZE (120) is divisible by 1, 2, 3, 4, 5, 6... very safe.
                
                qf, act_l, alph_l, qv = self._update_chunk_compiled(
                    obs_chunk, act_chunk, next_obs_chunk, rew_chunk, done_chunk,
                    CHUNK_SIZE
                )
                qf_loss_sum += qf
                actor_loss_sum += act_l
                alpha_loss_sum += alph_l
                q_val_sum += qv
                
                # Count actor updates in this chunk
                # Since we aligned start to 0 mod delay, and chunk is assumed multiple, 
                # count is just CHUNK_SIZE / delay
                actor_updates_count += (CHUNK_SIZE // self.policy_delay)
                
            else:
                # Use the individual compiled functions for the remainder
                for i in range(current_chunk):
                    # Slicing is fast (view)
                    obs, act, next_obs, rew, done = obs_chunk[i], act_chunk[i], next_obs_chunk[i], rew_chunk[i], done_chunk[i]
                    
                    global_update_idx = update_idx + i
                    if global_update_idx % self.policy_delay == 0:
                        qf, act_l, alph_l, qv, new_a = self._update_all_compiled(obs, act, next_obs, rew, done)
                        
                        qf_loss_sum += qf
                        actor_loss_sum += act_l
                        alpha_loss_sum += alph_l
                        q_val_sum += qv
                        actor_updates_count += 1
                    else:
                        qf, qv = self._update_critic_compiled(obs, act, next_obs, rew, done)
                        qf_loss_sum += qf
                        q_val_sum += qv

            step_time += (time.perf_counter() - t2_reshape)
            update_idx += current_chunk
            self.cumulative_model_updates += current_chunk
            
        t1_total = time.perf_counter()
        update_time = t1_total - t0_total

        # Sync metrics at the very end
        steps = updates_to_perform
        actor_steps = actor_updates_count.item()
        
        # If autotune, we need the latest alpha value.
        # We can just read it from the optimizer / log_alpha at the end.
        if self.autotune:
            self.alpha = self.log_alpha.exp().item()

        return {
            "Value Function Loss": (qf_loss_sum / steps).item(),
            "Policy Loss": (actor_loss_sum / max(actor_steps, 1)).item(),
            "Alpha": self.alpha,
            "Alpha Loss": (alpha_loss_sum / max(actor_steps, 1)).item(),
            "Mean Q Value": (q_val_sum / steps).item(),
            "Update Time": update_time,
            "Sample Time": sample_time,
            "Reshape Time": reshape_time,
            "Step Time": step_time,
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
