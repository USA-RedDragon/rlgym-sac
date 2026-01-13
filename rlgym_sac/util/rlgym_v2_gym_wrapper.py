import gymnasium as gym
import numpy as np


class RLGymV2GymWrapper(object):
    def __init__(self, rlgym_env):
        self.rlgym_env = rlgym_env
        self.agent_map = {}
        self.obs_buffer = np.zeros(1)
        print('WARNING: CALLING ENV.RESET() ONE EXTRA TIME TO DETERMINE STATE AND ACTION SPACES')
        obs_dict = rlgym_env.reset()
        obs_list = list(obs_dict.values())
        act_space = list(rlgym_env.action_spaces.values())[0] # [0] is agent_id, [1] is space? No, dict values are spaces.
        # rlgym v2 action_spaces is Dict[AgentID, SpaceType]
        
        # We need to peek at the spaces. 
        # Assuming homogeneous agents for now as that's typical for rlgym-sac
        any_agent = list(rlgym_env.agents)[0]
        act_space = rlgym_env.action_spaces[any_agent]
        obs_space = rlgym_env.observation_spaces[any_agent]

        self.is_discrete = False
        # RLGym V2 returns different space objects than gym. 
        # But commonly they have 'n' for discrete and 'shape' for box.
        # Or they might be actual gym spaces if the user configured it that way (unlikely in pure rlgym v2).
        
        # Let's check attributes
        if hasattr(act_space, 'n'):
            self.action_space = gym.spaces.Discrete(n=act_space.n)
            self.is_discrete = True
        elif hasattr(act_space, 'shape'):
             # Continuous
             self.action_space = gym.spaces.Box(low=-1, high=1, shape=act_space.shape)
        elif isinstance(act_space, tuple):
            if act_space[0] == 'discrete':
                self.action_space = gym.spaces.Discrete(n=act_space[1])
                self.is_discrete = True
            elif act_space[0] == 'continuous':
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(act_space[1],))
            else:
                self.action_space = None
        else:
             # Fallback or unknown
             self.action_space = None

        if hasattr(obs_space, 'shape'):
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_space.shape)
        else:
             if obs_list:
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=np.shape(obs_list[0]))
             else:
                self.observation_space = None

    def reset(self):
        self.agent_map.clear()
        
        # We need to correctly handle the reset return value from rlgym v2.
        # It returns Dict[AgentID, ObsType]
        obs_dict = self.rlgym_env.reset()
        
        # Also need to handle if it returns (obs, info) tuple which gym API expects, 
        # but RLGym v2 directly usually just returns obs_dict in reset().
        # However, if RLGym v2 follows Gym API strictly it might return (obs, info).
        # Let's check type.
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]

        idx = 0
        obs_vec = []
        
        for agent_id, agent_obs in obs_dict.items():
            self.agent_map[idx] = agent_id
            obs_vec.append(agent_obs)
            idx += 1
            
        if len(obs_vec) > 0:
             self.obs_buffer = np.asarray(obs_vec, dtype=np.float32)
        else:
             self.obs_buffer = np.zeros(1, dtype=np.float32)

        return self.obs_buffer

    def step(self, actions):
        if self.is_discrete:
            if len(actions.shape) > 1 and actions.shape[1] > 1:
                actions = np.argmax(actions, axis=1)
            actions = actions.astype(np.int32)

        action_dict = {}
        for i in range(len(actions)):
            if i in self.agent_map:
                agent_id = self.agent_map[i]
                action = actions[i]
                if self.is_discrete and not isinstance(action, np.ndarray):
                     action = np.array([action], dtype=np.int32)
                elif self.is_discrete and isinstance(action, np.ndarray) and action.ndim == 0:
                     action = action.reshape(1)
                
                action_dict[agent_id] = action

        obs_dict, reward_dict, terminated_dict, truncated_dict = self.rlgym_env.step(action_dict)

        rews = []
        done = False
        truncated = False
        idx = 0
        
        # Rebuild agent map if agents changed (e.g. dynamic team sizes)? 
        # Standard rlgym-sac assumes fixed number of agents per process mostly, but let's be safe.
        # The wrapper returns a flattened buffer, so we need consistent ordering.
        # The ordering comes from keys of obs_dict usually.
        
        # Reset buffer for new step
        obs_vec = []
        self.agent_map.clear()
        
        # We need to iterate over the returned observations to build the next state
        # In rlgym v2, step returns state for all current agents.
        for agent_id, agent_obs in obs_dict.items():
            self.agent_map[idx] = agent_id
            obs_vec.append(agent_obs)
            
            # Map rewards/done using agent_id
            rews.append(reward_dict.get(agent_id, 0.0))
            done = done or terminated_dict.get(agent_id, False)
            truncated = truncated or truncated_dict.get(agent_id, False)
            idx += 1

        self.obs_buffer = np.asarray(obs_vec)
        
        # Info is tricky, RLGym v2 doesn't pass a unified info dict, but state is accessible
        info = {"state": self.rlgym_env.state}
        return self.obs_buffer, rews, done, truncated, info

    def render(self):
        if hasattr(self.rlgym_env, 'render'):
            return self.rlgym_env.render()

    def close(self):
        if hasattr(self.rlgym_env, 'close'):
            return self.rlgym_env.close()
