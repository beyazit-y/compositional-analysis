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
from stable_baselines3.common.utils import set_random_seed
from metadrive.component.map.pg_map import MapGenerateMethod
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

def train_env():
    config = dict(
        map=4,
        discrete_action=True,
        discrete_throttle_dim=3,
        discrete_steering_dim=3,
        horizon=1000,
        random_spawn_lane_index=True,
        num_scenarios=1000,
        start_seed=1000,
        traffic_density=0.1,
        need_inverse_traffic=True,
        accident_prob=0.1,
        log_level=50,
        random_lane_width=True,
        random_agent_model=True,
        random_lane_num=True,
    )
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
    args = parser.parse_args()

    # while True:
    #     env=train_env()
    #     temp = env.reset()
    #     print(temp)
    #     ret = env.render(mode="topdown", 
    #                      window=False,
    #                      screen_size=(600, 600), 
    #                      camera_position=(50, 50))
    #     env.close()
    #     plt.axis("off")
    #     plt.imshow(ret)
    #     plt.show()
    #     clear_output()

    set_random_seed(args.seed)
    n_envs = 16
    env = SubprocVecEnv([partial(train_env) for _ in range(n_envs)])
    model = PPO("MlpPolicy", 
                env=env,
                n_steps=4096,
                verbose=1)
    model.learn(total_timesteps=1_000_000,
                log_interval=1)
    env.close()
    clear_output()

    save_path = os.path.join(args.save_dir, f"model.zip")
    model.save(save_path)
    print("Training is finished.")

