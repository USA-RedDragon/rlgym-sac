"""
File: learner.py
Author: Matthew Allen

Description:
The primary algorithm file. The Learner object coordinates timesteps from the workers 
and sends them to SAC, keeps track of the misc. variables and statistics for logging,
reports to wandb and the console, and handles checkpointing.
"""

import json
import os
import random
import shutil
import time
from typing import Union

import numpy as np
import torch
import wandb
from wandb.wandb_run import Run

from rlgym_sac.batched_agents import BatchedAgentManager
from rlgym_sac.sac import SACLearner
from rlgym_sac.util import WelfordRunningStat, reporting, KBHit


class Learner(object):
    def __init__(
            self,
            env_create_function,
            metrics_logger=None,
            n_proc: int = 8,
            min_inference_size: int = 80,
            render: bool = False,
            render_delay: float = 0,

            timestep_limit: int = 5_000_000_000,
            exp_buffer_size: int = 1_000_000,
            ts_per_iteration: int = 50000,
            standardize_returns: bool = True,
            standardize_obs: bool = True,
            max_returns_per_stats_increment: int = 150,
            steps_per_obs_stats_increment: int = 5,

            # SAC Specific args
            sac_batch_size: int = 256,
            sac_ent_coef: Union[str, float] = 'auto',
            sac_learning_rate: float = 3e-4,
            sac_learning_starts: int = 100,
            sac_train_freq: int = 1,
            sac_gradient_steps: int = 1,
            sac_tau: float = 0.005,
            sac_gamma: float = 0.99,
            policy_layer_sizes: tuple = (256, 256),
            critic_layer_sizes: tuple = (256, 256),

            log_to_wandb: bool = False,
            load_wandb: bool = True,
            wandb_run: Union[Run, None] = None,
            wandb_project_name: Union[str, None] = None,
            wandb_group_name: Union[str, None] = None,
            wandb_run_name: Union[str, None] = None,

            checkpoints_save_folder: Union[str, None] = None,
            add_unix_timestamp: bool = True,
            checkpoint_load_folder: Union[str, None] = "latest", # "latest" loads latest checkpoint
            save_every_ts: int = 1_000_000,

            instance_launch_delay: Union[float, None] = None,
            random_seed: int = 123,
            n_checkpoints_to_keep: int = 5,
            shm_buffer_size: int = 8192,
            device: str = "auto",
            use_amp: bool = True):

        assert (
                env_create_function is not None
        ), "MUST PROVIDE A FUNCTION TO CREATE RLGYM FUNCTIONS TO INITIALIZE RLGYM-PPO"

        if checkpoints_save_folder is None:
            checkpoints_save_folder = os.path.join(
                "data", "checkpoints", "rlgym-sac-run"
            )

        # Add the option for the user to turn off the addition of Unix Timestamps to
        # the ``checkpoints_save_folder`` path
        self.add_unix_timestamp = add_unix_timestamp
        if add_unix_timestamp:
            checkpoints_save_folder = f"{checkpoints_save_folder}-{time.time_ns()}"

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.n_checkpoints_to_keep = n_checkpoints_to_keep
        self.checkpoints_save_folder = checkpoints_save_folder
        self.max_returns_per_stats_increment = max_returns_per_stats_increment
        self.metrics_logger = metrics_logger
        self.standardize_returns = standardize_returns
        self.save_every_ts = save_every_ts
        self.ts_since_last_save = 0

        if device in {"auto", "gpu"} and torch.cuda.is_available():
            self.device = "cuda:0"
            torch.backends.cudnn.benchmark = True
        elif device == "auto" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device
        
        print(f"Using device {self.device}")
        self.exp_buffer_size = exp_buffer_size
        self.timestep_limit = timestep_limit
        self.ts_per_epoch = ts_per_iteration
        self.return_stats = WelfordRunningStat(1)
        self.epoch = 0
        self.use_amp = use_amp

        print("Initializing processes...")
        collect_metrics_fn = None if metrics_logger is None else self.metrics_logger.collect_metrics
        self.agent = BatchedAgentManager(
            None, min_inference_size=min_inference_size,
            seed=random_seed,
            standardize_obs=standardize_obs,
            steps_per_obs_stats_increment=steps_per_obs_stats_increment
        )
        obs_space_size, act_space_size = self.agent.init_processes(
            n_processes=n_proc,
            build_env_fn=env_create_function,
            collect_metrics_fn=collect_metrics_fn,
            spawn_delay=instance_launch_delay,
            render=render,
            render_delay=render_delay,
            shm_buffer_size=shm_buffer_size
        )
        obs_space_size = np.prod(obs_space_size)
        print("Initializing SAC...")

        self.sac_learner = SACLearner(
            obs_space_size=obs_space_size,
            act_space_size=act_space_size,
            device=self.device,
            batch_size=sac_batch_size,
            ent_coef=sac_ent_coef,
            learning_rate=sac_learning_rate,
            buffer_size=exp_buffer_size,
            learning_starts=sac_learning_starts,
            train_freq=sac_train_freq,
            gradient_steps=sac_gradient_steps,
            tau=sac_tau,
            gamma=sac_gamma,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            use_amp=self.use_amp,
        )

        self.agent.policy = self.sac_learner.policy

        self.config = {
            "n_proc": n_proc,
            "min_inference_size": min_inference_size,
            "timestep_limit": timestep_limit,
            "exp_buffer_size": exp_buffer_size,
            "ts_per_iteration": ts_per_iteration,
            "standardize_returns": standardize_returns,
            "standardize_obs": standardize_obs,
            "sac_batch_size": sac_batch_size,
            "sac_ent_coef": sac_ent_coef,
            "sac_learning_rate": sac_learning_rate,
            "sac_learning_starts": sac_learning_starts,
            "sac_train_freq": sac_train_freq,
            "sac_gradient_steps": sac_gradient_steps,
            "sac_tau": sac_tau,
            "sac_gamma": sac_gamma,
            "shm_buffer_size": shm_buffer_size,
            "use_amp": use_amp,
        }

        self.wandb_run = wandb_run
        wandb_loaded = checkpoint_load_folder is not None and self.load(checkpoint_load_folder, load_wandb)

        if log_to_wandb and self.wandb_run is None and not wandb_loaded:
            project = "rlgym-sac" if wandb_project_name is None else wandb_project_name
            group = "unnamed-runs" if wandb_group_name is None else wandb_group_name
            run_name = "rlgym-sac-run" if wandb_run_name is None else wandb_run_name
            print("Attempting to create new wandb run...")
            self.wandb_run = wandb.init(
                project=project, group=group, config=self.config, name=run_name, reinit=True
            )
            print("Created new wandb run!", self.wandb_run.id)
        print("Learner successfully initialized!")

    def learn(self):
        """
        Function to wrap the _learn function in a try/catch/finally
        block to ensure safe execution and error handling.
        :return: None
        """
        try:
            self._learn()
        except Exception:
            import traceback

            print("\n\nLEARNING LOOP ENCOUNTERED AN ERROR\n")
            traceback.print_exc()

            try:
                self.save(self.agent.cumulative_timesteps)
            except:
                print("FAILED TO SAVE ON EXIT")

        finally:
            self.cleanup()

    def _learn(self):
        """
        Learning function.
        :return: None
        """

        # Class to watch for keyboard hits; may fail when no TTY (e.g., spawned sweep)
        try:
            kb = KBHit()
            print("Press (p) to pause (c) to checkpoint, (q) to checkpoint and quit (after next iteration)\n")
        except Exception as e:
            kb = None
            print(f"KBHit disabled (no TTY): {e}")

        while self.agent.cumulative_timesteps < self.timestep_limit:
            epoch_start = time.perf_counter()
            report = {}

            # Collect the desired number of timesteps from our agent.
            experience, collected_metrics, steps_collected, collection_time = self.agent.collect_timesteps(
                self.ts_per_epoch
            )

            if self.metrics_logger is not None:
                self.metrics_logger.report_metrics(collected_metrics, self.wandb_run, self.agent.cumulative_timesteps)

            # Add experience to SAC buffer
            self.sac_learner.add_experience(experience)
            
            if self.standardize_returns:
                 # Standardize returns Logic for reporting ONLY (not affecting training buffer for SAC)
                 # experience is (states, actions, log_probs, rewards, next_states, dones, truncated)
                 rewards = experience[3]
                 batch_rewards = rewards.flatten() # assuming numpy
                 # Just increment stats
                 n_to_increment = min(self.max_returns_per_stats_increment, len(batch_rewards))
                 self.return_stats.increment(batch_rewards[:n_to_increment], n_to_increment)

            # Train SAC
            # We collected 'steps_collected' new steps.
            # Typically we do 1 gradient step per env step.
            # So steps_to_train = steps_collected
            
            train_start = time.perf_counter()
            sac_report = self.sac_learner.learn(steps_collected=steps_collected)
            train_end = time.perf_counter()
            train_time = train_end - train_start

            epoch_stop = time.perf_counter()
            epoch_time = epoch_stop - epoch_start
            
            # Report variables
            if sac_report:
                report.update(sac_report)

            report["Cumulative Timesteps"] = self.agent.cumulative_timesteps
            report["Total Iteration Time"] = epoch_time
            report["Timesteps Collected"] = steps_collected
            report["Timestep Collection Time"] = collection_time
            report["Timestep Consumption Time"] = train_time
            report["Collected Steps per Second"] = steps_collected / collection_time
            report["Overall Steps per Second"] = steps_collected / epoch_time

            self.ts_since_last_save += steps_collected
            if self.agent.average_reward is not None:
                report["Policy Reward"] = self.agent.average_reward
            else:
                report["Policy Reward"] = np.nan

            # Log to wandb and print to the console.
            reporting.report_metrics(loggable_metrics=report,
                                     debug_metrics=None,
                                     wandb_run=self.wandb_run)

            report.clear()
            
            if "cuda" in self.device:
                torch.cuda.empty_cache()

            # Check keyboard
            if kb is not None and kb.kbhit():
                c = kb.getch()
                if c == 'p':  # pause
                    print("Paused, press any key to resume")
                    while True:
                        if kb.kbhit():
                            break
                if c in ('c', 'q'):
                    self.save(self.agent.cumulative_timesteps)
                if c == 'q':
                    return
                if c in ('c', 'p'):
                    print("Resuming...\n")

            # Save if we've reached the next checkpoint timestep.
            if self.ts_since_last_save >= self.save_every_ts:
                self.save(self.agent.cumulative_timesteps)
                self.ts_since_last_save = 0

            self.epoch += 1

    def save(self, cumulative_timesteps):
        """
        Function to save a checkpoint.
        :param cumulative_timesteps: Number of timesteps that have passed so far.
        :return: None
        """

        folder_path = os.path.join(
            self.checkpoints_save_folder, str(cumulative_timesteps)
        )
        os.makedirs(folder_path, exist_ok=True)

        print(f"Saving checkpoint {cumulative_timesteps}...")
        existing_checkpoints = [
            int(arg) for arg in os.listdir(self.checkpoints_save_folder)
        ]
        if len(existing_checkpoints) > self.n_checkpoints_to_keep:
            existing_checkpoints.sort()
            for checkpoint_name in existing_checkpoints[: -self.n_checkpoints_to_keep]:
                shutil.rmtree(
                    os.path.join(self.checkpoints_save_folder, str(checkpoint_name))
                )

        os.makedirs(folder_path, exist_ok=True)

        self.sac_learner.save_to(folder_path)

        book_keeping_vars = {
            "cumulative_timesteps": self.agent.cumulative_timesteps,
            "cumulative_model_updates": self.sac_learner.cumulative_model_updates,
            "policy_average_reward": self.agent.average_reward,
            "epoch": self.epoch,
            "ts_since_last_save": self.ts_since_last_save,
            "reward_running_stats": self.return_stats.to_json(),
        }
        if self.agent.standardize_obs:
            book_keeping_vars["obs_running_stats"] = self.agent.obs_stats.to_json()
        
        if self.wandb_run is not None:
            book_keeping_vars["wandb_run_id"] = self.wandb_run.id
            book_keeping_vars["wandb_project"] = self.wandb_run.project
            book_keeping_vars["wandb_entity"] = self.wandb_run.entity
            book_keeping_vars["wandb_group"] = self.wandb_run.group
            book_keeping_vars["wandb_config"] = self.wandb_run.config.as_dict()

        book_keeping_table_path = os.path.join(folder_path, "BOOK_KEEPING_VARS.json")
        with open(book_keeping_table_path, "w") as f:
            json.dump(book_keeping_vars, f, indent=4)

        print(f"Checkpoint {cumulative_timesteps} saved!\n")

    def load(self, folder_path, load_wandb):
        """
        Function to load from a checkpoint.
        """

        if folder_path == "latest":
            save_folder = self.checkpoints_save_folder
            if save_folder is None:
                return

            if self.add_unix_timestamp:
                base_save_folder = save_folder[:save_folder.rfind('-')]
                save_path = os.path.dirname(base_save_folder)

                if not os.path.exists(save_path):
                    return

                highest_timestamp = -1
                best_folder = None
                for filename in os.listdir(save_path):
                    full_path = os.path.join(save_path, filename)
                    if not os.path.isdir(full_path):
                        continue

                    if full_path.startswith(base_save_folder):
                        unix_start_idx = full_path.rfind('-') + 1
                        if unix_start_idx > 0:
                            unix_time_str = filename[unix_start_idx:]
                            if unix_time_str.isdigit():
                                timestamp = int(unix_time_str)
                                if timestamp > highest_timestamp:
                                    highest_timestamp = timestamp
                                    best_folder = full_path

                if not (best_folder is None):
                    load_base_path = best_folder
                else:
                    return
            else:
                if os.path.exists(self.checkpoints_save_folder):
                    load_base_path = self.checkpoints_save_folder
                else:
                    return

            highest_ts = -1
            for filename in os.listdir(load_base_path):
                if not os.path.isdir(os.path.join(load_base_path, filename)):
                    continue

                if not filename.isdigit():
                    continue

                highest_ts = max(highest_ts, int(filename))

            if highest_ts != -1:
                folder_path = os.path.join(load_base_path, str(highest_ts))
                print(f"Auto-load path: {folder_path}")
            else:
                return

        assert os.path.exists(folder_path), f"UNABLE TO LOCATE FOLDER {folder_path}"
        print(f"Loading from checkpoint at {folder_path}")

        self.sac_learner.load_from(folder_path)

        wandb_loaded = False
        with open(os.path.join(folder_path, "BOOK_KEEPING_VARS.json"), "r") as f:
            book_keeping_vars = dict(json.load(f))
            self.agent.cumulative_timesteps = book_keeping_vars["cumulative_timesteps"]
            self.agent.average_reward = book_keeping_vars["policy_average_reward"]
            self.sac_learner.cumulative_model_updates = book_keeping_vars.get("cumulative_model_updates", 0)
            
            if "reward_running_stats" in book_keeping_vars:
                self.return_stats.from_json(book_keeping_vars["reward_running_stats"])

            if self.agent.standardize_obs and "obs_running_stats" in book_keeping_vars.keys():
                self.agent.obs_stats = WelfordRunningStat(1)
                self.agent.obs_stats.from_json(book_keeping_vars["obs_running_stats"])

            self.epoch = book_keeping_vars.get("epoch", 0)
            
            if "wandb_run_id" in book_keeping_vars and load_wandb:
                self.wandb_run = wandb.init(
                    settings=wandb.Settings(start_method="spawn"),
                    entity=book_keeping_vars["wandb_entity"],
                    project=book_keeping_vars["wandb_project"],
                    group=book_keeping_vars["wandb_group"],
                    id=book_keeping_vars["wandb_run_id"],
                    config=book_keeping_vars["wandb_config"],
                    resume="allow",
                    reinit=True,
                )
                wandb_loaded = True

        print("Checkpoint loaded!")
        return wandb_loaded

    def cleanup(self):
        """
        Function to clean everything up before shutting down.
        :return: None.
        """

        if self.wandb_run is not None:
            self.wandb_run.finish()
        if type(self.agent) == BatchedAgentManager:
            self.agent.cleanup()
