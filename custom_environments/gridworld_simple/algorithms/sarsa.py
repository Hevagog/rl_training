import numpy as np


class Sarsa:
    def __init__(
        self,
        gamma,
        alpha,
        action_space,
        observation_space,
        epsilon_0=1.0,
        epsilon_min=0.01,
        decay_rate=0.001,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.action_space = action_space
        self.observation_space = observation_space
        self.q_table = np.zeros((self.observation_space, self.action_space))
        self.time_step = 0
        self.epsilon_0 = epsilon_0
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.epsilon = self.epsilon_0

    def update(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action] if next_state is not None else 0

        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        return new_q

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def _update_epsilon(self):
        self.time_step += 1
        self.epsilon = self.epsilon_min + (self.epsilon_0 - self.epsilon_min) * np.exp(
            -self.decay_rate * self.time_step
        )

    def train(self, env, episodes, plot_every=100, decay_epsilon=False):
        score_history = []
        temp_history = []
        for episode in range(episodes):

            episode_score = 0
            state = env.reset()[0]
            action = self.get_action(state)

            if decay_epsilon:
                self._update_epsilon()
            else:
                self.epsilon = 1 / (episode + 1)

            done = False
            while True:
                next_state, reward, done, _, _ = env.step(action)
                episode_score += reward
                if not done:
                    next_action = self.get_action(next_state)

                    self.q_table[state][action] = self.update(
                        state, action, reward, next_state, next_action
                    )
                    state = next_state
                    action = next_action
                else:
                    self.q_table[state][action] = self.update(
                        state, action, reward, None, None
                    )
                    temp_history.append(episode_score)
                    break
            if episode % plot_every == 0:
                score_history.append(np.mean(temp_history))
                print(f"Episode: {episode}, Score: {np.mean(temp_history)}")
                temp_history = []

        return score_history
