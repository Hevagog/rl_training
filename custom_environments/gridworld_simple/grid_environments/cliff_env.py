import numpy as np

from .gridenv import GridEnv


class CliffEnv(GridEnv):
    def __init__(self, size_x, size_y, window_shape=512, render_mode=None):
        super().__init__(size_x, size_y, window_shape, render_mode)
        self._target_location = np.array([self.size_x - 1, 0], dtype=np.int64)
        self._agent_location = np.array([0, 0], dtype=np.int64)
        self._cliff = np.zeros((self.size_x, self.size_y), dtype=bool)
        self._cliff[1:-1, 0] = True

    def reset(self, seed=None, options=None):
        self._agent_location = np.array([0, 0], dtype=np.int64)
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(negative_reward=self._cliff)

        return observation, info

    def step(self, action):
        # Map action ( element of {0, 1, 2, 3} ) to direction agent will move.
        direction = self._action_to_direction[action]
        reward = 0
        # np.clip ensures the agent stays within the grid.
        self._agent_location = np.clip(
            self._agent_location + direction,
            np.array([0, 0]),
            np.array([self.size_x - 1, self.size_y - 1]),
        )
        # Agent falls off the cliff.
        if self._cliff[tuple(self._agent_location)]:
            reward = -100
            terminated = True
        else:
            # Episode ends when agent reaches the target.
            terminated = np.array_equal(self._agent_location, self._target_location)
            if terminated:
                reward = 10
            else:
                reward = -1
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(negative_reward=self._cliff)

        return observation, reward, terminated, False, info
