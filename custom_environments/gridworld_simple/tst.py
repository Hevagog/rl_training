from time import sleep
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

from grid_environments.windy_gridworld import WindyGridWOrldEnv
from algorithms.sarsa import Sarsa
from grid_environments.cliff_env import CliffEnv


if __name__ == "__main__":
    # env = WindyGridWOrldEnv(render_mode="human", size_x=6, size_y=5)
    # env = CliffEnv(render_mode="rgb_array", size_x=6, size_y=5)
    env = gym.make("CliffWalking-v0")
    # print(env.reset())
    # model = DQN("MultiInputPolicy", env, verbose=1)
    # model.learn(total_timesteps=20_000)
    # # eval_env = gym.make("CliffWalking-v0")
    # eval_env = CliffEnv(render_mode="human", size_x=6, size_y=5)
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    size_x = 6
    size_y = 5
    # env = CliffEnv(render_mode="rgb_array", size_x=size_x, size_y=size_y)

    agent = Sarsa(
        gamma=0.9,
        alpha=0.1,
        epsilon_0=1.0,
        epsilon_min=0.01,
        decay_rate=0.0005,
        action_space=env.action_space.n,
        observation_space=env.observation_space.n,
    )
    history = agent.train(env, episodes=100_000, decay_epsilon=True, plot_every=1000)

    policy_sarsa = np.array(
        [
            np.argmax(agent.q_table[key]) if key in agent.q_table else -69
            for key in np.arange(48)
        ]
    ).reshape(4, 12)
    print(policy_sarsa)

    # eval_env = CliffEnv(render_mode="human", size_x=size_x, size_y=size_y)

    # state = eval_env.reset()[0]
    # done = False
    # while not done:
    #     sleep(1)
    #     action = agent.get_action(state)
    #     observation, reward, done, _, _ = env.step(action)
    #     state = observation
    #     eval_env.render()

    # save history to csv
    # np.savetxt("sarsa_history.csv", history, delimiter=",")
