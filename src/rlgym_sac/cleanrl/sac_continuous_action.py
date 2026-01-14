# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, layer_sizes=(256, 256)):
        super().__init__()
        # Direct implementation for common case to help Compiler/Dynamo
        if layer_sizes == (256, 256):
            input_dim = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 1)
            self.net_arch = 2
            self.use_seq = False
        elif layer_sizes == (256, 256, 256):
            input_dim = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, 1)
            self.net_arch = 3
            self.use_seq = False
        else:
            layers = []
            input_dim = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)
            
            for size in layer_sizes:
                layers.append(nn.Linear(input_dim, size))
                layers.append(nn.ReLU())
                input_dim = size
                
            layers.append(nn.Linear(input_dim, 1))
            self.net = nn.Sequential(*layers)
            self.use_seq = True

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        if not self.use_seq:
            if self.net_arch == 2:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
            elif self.net_arch == 3:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = self.fc4(x)
                return x
        else:
            return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, layer_sizes=(256, 256)):
        super().__init__()
        
        # Direct implementation for common case to help Compiler/Dynamo
        if layer_sizes == (256, 256):
            input_dim = np.array(env.single_observation_space.shape).prod()
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
            self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
            self.net_arch = 2
            self.use_seq = False
        elif layer_sizes == (256, 256, 256):
            input_dim = np.array(env.single_observation_space.shape).prod()
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
            self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
            self.net_arch = 3
            self.use_seq = False
        else:
            layers = []
            input_dim = np.array(env.single_observation_space.shape).prod()
            
            for size in layer_sizes:
                layers.append(nn.Linear(input_dim, size))
                layers.append(nn.ReLU())
                input_dim = size
                
            self.net = nn.Sequential(*layers)
            
            self.fc_mean = nn.Linear(input_dim, np.prod(env.single_action_space.shape))
            self.fc_logstd = nn.Linear(input_dim, np.prod(env.single_action_space.shape))
            self.use_seq = True

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        if not self.use_seq:
            if self.net_arch == 2:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
            elif self.net_arch == 3:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
        else:
            x = self.net(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
