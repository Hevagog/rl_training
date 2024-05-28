import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
import pygame


class GridEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 10}
    _agent_location = None
    _target_location = None

    def __init__(self, size_x, size_y, window_shape=512, render_mode=None):
        self.size_x = size_x
        self.size_y = size_y

        self.window_width = window_shape * self.size_x // max(self.size_x, self.size_y)
        self.window_height = window_shape * self.size_y // max(self.size_x, self.size_y)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.size_x, self.size_y]),
                    shape=(2,),
                    dtype=int,
                ),
                "target": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.size_x, self.size_y]),
                    shape=(2,),
                    dtype=int,
                ),
            }
        )

        self.action_space = spaces.Discrete(4)
        """
        0: Move right
        1: Move up
        2: Move left
        3: Move down
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render.modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_observation(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_width // self.size_x
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for y in range(self.size_y + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * y),
                (self.window_width, pix_square_size * y),
                width=3,
            )
        for x in range(self.size_x + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_height),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
