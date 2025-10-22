import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import csv
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
from metadrive.utils.draw_top_down_map import draw_top_down_map

def test_env(scenario):
    config = dict(
        map=scenario,
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
        vehicle_config={
            "spawn_velocity": [10.0, 0.0], # m/s; default max_speed_km_h is 80 km/h
            "spawn_velocity_car_frame": True
        }
    )
    return MetaDriveEnv(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test policy in MetaDrive")
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
        "--model",
        type=str,
        default="storage/model.zip",
        help="Saved model zip")
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of test samples")
    parser.add_argument(
        "--scenario",
        type=str,
        default="XX",
        help="Scenario string")
    args = parser.parse_args()

    # while True:
    #     env=test_env(args.scenario)
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

    model = PPO.load(args.model)

    env = test_env(args.scenario)

    all_traces = []
    csv_path = os.path.join(args.save_dir, "traces.csv")
    trace_id = 0

    with open(csv_path, "w", newline="") as f:
        writer = None

        for ep in range(args.n):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            step = 0

            print(f"\n=== Episode {ep+1}/{args.n} ===")
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward

                vehicle = env.vehicle
                pos = vehicle.position
                heading = vehicle.heading_theta
                vel = vehicle.speed

                row = {
                    "trace_id": trace_id,
                    "step": step,
                    "x": pos[0],
                    "y": pos[1],
                    "heading": heading,
                    "speed": vel,
                    "action": action.tolist() if hasattr(action, "tolist") else action,
                    "reward": reward,
                }

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    writer.writeheader()

                writer.writerow(row)
                step += 1

                env.render(
                    mode="topdown",
                    screen_record=True,
                    window=False
                )

            print(f"Episode reward: {total_reward:.2f}")

            gif_path = os.path.join(args.save_dir, f"trace_{trace_id:03d}.gif")
            env.top_down_renderer.generate_gif(gif_path)
            print(f"Saved gif to {gif_path}")

            trace_id += 1

    env.close()
    print(f"\nAll {args.n} traces saved to {csv_path}")

