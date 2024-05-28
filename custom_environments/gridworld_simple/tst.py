from time import sleep

from grid_environments.windy_gridworld import WindyGridWOrldEnv


if __name__ == "__main__":
    env = WindyGridWOrldEnv(render_mode="human", size_x=6, size_y=10)
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
        sleep(0.2)
