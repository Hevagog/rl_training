import numpy as np
import tqdm


class NStepSarsa:
    def __init__(
        self,
        gamma,
        alpha,
        action_space,
        observation_space,
        epsilon_0=1.0,
        epsilon_min=0.01,
        decay_rate=0.001,
        n=1,
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
        self.n = n

    def update(self, state, action, G):
        current_q = self.q_table[state][action]
        self.q_table[state][action] += self.alpha * (G - current_q)

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def _update_epsilon(self, episode):
        self.epsilon = max(
            self.epsilon_min, self.epsilon_0 * np.exp(-self.decay_rate * episode)
        )

    def train(self, env, episodes, plot_every=100, decay_epsilon=True):
        score_history = []
        temp_history = []

        for episode in tqdm.tqdm(range(episodes)):
            episode_score = 0
            state = env.reset()[0]
            action = self.get_action(state)
            n_step_buffer = []
            state_buffer = [state]
            action_buffer = [action]

            if decay_epsilon:
                self._update_epsilon(episode)
            else:
                self.epsilon = 1 / (episode + 1)

            done = False
            tau = 0
            t = 0
            T = np.inf
            while tau < T - 1:
                if t < T:
                    next_state, reward, done, _, _ = env.step(action)
                    episode_score += reward
                    n_step_buffer.append(reward)
                    state_buffer.append(next_state)
                    if done:
                        T = t + 1
                    else:
                        next_action = self.get_action(next_state)
                        action_buffer.append(next_action)
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau, min(tau + self.n, T)):
                        G += (self.gamma ** (i - tau)) * n_step_buffer[i - tau]
                    if tau + self.n < T:
                        G += (
                            self.gamma**self.n
                            * self.q_table[state_buffer[tau + self.n]][
                                action_buffer[tau + self.n]
                            ]
                        )
                    state_t = state_buffer[tau]
                    action_t = action_buffer[tau]
                    self.update(state_t, action_t, G)
                if tau + 1 >= T:
                    break
                if not done:
                    state = next_state
                    action = next_action
                t += 1
            score_history.append(episode_score)
            if episode % plot_every == 0:
                temp_history.append(np.mean(score_history))
                print(
                    f"Episode {episode}, Average Score: {np.mean(score_history[-plot_every:])}"
                )
        return temp_history
