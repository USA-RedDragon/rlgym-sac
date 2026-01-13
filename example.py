import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym_sac.util import MetricsLogger
from rlgym.api import RLGym
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
from rlgym.rocket_league import common_values
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward
from typing import Dict, Any, List
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM, BACK_WALL_Y

class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
     def __init__(self, own_goal=False):
         self.own_goal = own_goal

     def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
         pass

     def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool], is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
         rewards = {}
         ball_vel = state.ball.linear_velocity
         
         for agent in agents:
             car = state.cars[agent]
             if car.team_num == BLUE_TEAM:
                 objective = np.array([0, BACK_WALL_Y, 0])
             else:
                 objective = np.array([0, -BACK_WALL_Y, 0])
             
             if self.own_goal:
                 objective = -objective
                 
             # Velocity of ball towards goal
             # Projected velocity: vel . dir
             ball_to_goal = objective - state.ball.position
             dist = np.linalg.norm(ball_to_goal)
             if dist > 0:
                 dir_to_goal = ball_to_goal / dist
                 rewards[agent] = float(np.dot(ball_vel, dir_to_goal)) / 2300.0 # Normalize by approx max speed
             else:
                 rewards[agent] = 0.0
         return rewards

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState, done: bool) -> list:
        # Assuming single agent for simple example or taking first
        if not game_state.cars:
            return [np.zeros(3), np.eye(3), 0]
        
        agent_id = list(game_state.cars.keys())[0]
        car = game_state.cars[agent_id]
        
        # Scoring is tricky in v2, assume based on goal scored event or track manually
        score = 0 # Placeholder as GameState doesn't track score directly in v2 typically without wrapper or stateful component
        
        return [car.physics.linear_velocity,
                car.physics.rotation_mtx,
                score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        if len(collected_metrics) > 0:
            avg_linvel /= len(collected_metrics)
        
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupTableAction()
    terminal_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout_ticks)

    reward_fn = CombinedReward(
        (VelocityPlayerToBallReward(), 0.01),
        (VelocityBallToGoalReward(), 0.1),
        (GoalReward(), 10.0),
        (TouchReward(), 0.1) # Demo reward replacement if needed, using touch as placeholder or find demo reward
    )

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    # Note: State mutator needed for spawning cars usually
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    
    # We need to define team sizes. 
    # If spawn_opponents is True, we likely want 1v1.
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size if spawn_opponents else 0),
        KickoffMutator()
    )

    from rlgym.rocket_league.rlviser import RLViserRenderer
    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=terminal_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer()
    )

    return env


if __name__ == "__main__":
    from rlgym_sac import Learner
    metrics_logger = ExampleLogger()

    # 32 processes
    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      render=True,
                      ts_per_iteration=50000,
                      exp_buffer_size=1_000_000,
                      sac_batch_size=256,
                      sac_ent_coef='auto',
                      sac_learning_rate=3e-4,
                      sac_learning_starts=10000,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=1_000_000_000,
                      log_to_wandb=True)
    learner.learn()