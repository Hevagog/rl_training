import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
import pygame

from .gridenv import GridEnv


class WindyGridWOrldEnv(GridEnv):
    def __init__(self, render_mode=None, size_x=5, size_y=5, wind=None):
        super().__init__(render_mode=render_mode, size_x=size_x, size_y=size_y)
        self.wind = wind

    def reset(self, seed=None, options=None):
        self.wind = self.np_random.choice([-1, 0, 1], size=(self.size_x,))
        self._agent_location = self.np_random.integers(
            low=np.array([0, 0]),
            high=np.array([self.size_x, self.size_y]),
            size=2,
            dtype=int,
        )
        self._target_location = self.np_random.integers(
            low=np.array([0, 0]),
            high=np.array([self.size_x, self.size_y]),
            size=2,
            dtype=int,
        )

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                low=np.array([0, 0]),
                high=np.array([self.size_x, self.size_y]),
                size=2,
                dtype=int,
            )

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map action ( element of {0, 1, 2, 3} ) to direction agent will move.
        direction = self._action_to_direction[action]
        # np.clip ensures the agent stays within the grid.
        self._agent_location = np.clip(
            self._agent_location + direction,
            np.array([0, 0]),
            np.array([self.size_x, self.size_y]),
        )
        # Episode ends when agent reaches the target.
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, reward, False, info
