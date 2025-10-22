import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import argparse
import gymnasium as gym
from functools import partial
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from metadrive.envs import MetaDriveEnv
from IPython.display import Image, clear_output
from metadrive.utils.doc_utils import generate_gif
from metadrive.component.map.base_map import BaseMap
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from metadrive.component.map.pg_map import MapGenerateMethod
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map


def train_env(monitor=True):
    config = dict(
        map=2,
        discrete_action=True,
        horizon=3000,
        random_spawn_lane_index=True,
        num_scenarios=1000,
        start_seed=1000,
        traffic_density=0.05,
        need_inverse_traffic=True,
        accident_prob=0.0,
        log_level=50,
        random_lane_width=False,
        random_agent_model=False,
        random_lane_num=True,
    )
    if monitor:
        return Monitor(MetaDriveEnv(config))
    else:
        return MetaDriveEnv(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train policy in MetaDrive")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="storage",
        help="Directory to save the trained model")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=16,
        help="Number of parallel environments")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Number of environment steps")
    args = parser.parse_args()

    # while True:
    #     env=train_env(monitor=False)
    #     env.reset()
    #     ret = draw_top_down_map(env.current_map)
    #     # ret = env.render(mode="topdown", window=False)
    #     # ret = env.render(mode="topdown",
    #     #                  window=False,
    #     #                  # screen_size=(600, 600),
    #     #                  # camera_position=(50, 50)
    #     #                  )
    #     env.close()
    #     plt.axis("off")
    #     plt.imshow(ret)
    #     plt.show()
    #     clear_output()

    set_random_seed(args.seed)
    env = SubprocVecEnv([partial(train_env) for _ in range(args.n_envs)])
    model = PPO("MlpPolicy", 
                env=env,
                n_steps=4096,
                verbose=1)
    model.learn(total_timesteps=args.timesteps,
                log_interval=1)
    env.close()
    clear_output()

    save_path = os.path.join(args.save_dir, f"model.zip")
    model.save(save_path)
    print("Training is finished.")

