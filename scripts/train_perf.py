import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="models/perf_cartpole_ppo.zip")
    parser.add_argument("--timesteps", type=int, default=100_000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.output)
    print(f"saved PPO model to {args.output}")


if __name__ == "__main__":
    main()
