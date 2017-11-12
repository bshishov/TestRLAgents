import numpy as np

from . import agent


class QLearningAgent(agent.Agent):
    Q = None
    state = None

    def __init__(self, state_space, action_space, learning_rate=0.8, discount_factor=0.95):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = np.zeros([state_space, action_space])

    def episode_start(self, initial_state, state_space, action_space):
        self.state = initial_state
        self.action_space = action_space
        pass

    def step(self, state, reward, done, available_actions):
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + self.learning_rate * (reward + self.discount_factor * np.max(self.Q[state, :]) - self.Q[self.state, self.action])
        rAll += r

        self.state = state
        action = np.argmax(self.Q[self.state, :] + np.random.randn(1, self.action_space) * (1. / (i + 1)))

        return action

    def episode_end(self):
        pass