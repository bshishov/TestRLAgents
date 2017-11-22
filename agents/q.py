from agents import Agent
import numpy as np

from agents.fn import TableQFunction


class QAgent(Agent):
    Q = None
    state = None

    def __init__(self, q=TableQFunction(), learning_rate=0.8, discount_factor=0.95):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = q

    def step(self, state, reward, available_actions, done):
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + self.learning_rate * (reward + self.discount_factor * np.max(self.Q[state, :]) - self.Q[self.state, self.action])
        rAll += r

        self.state = state
        action = np.argmax(self.Q[self.state, :] + np.random.randn(1, self.action_space) * (1. / (i + 1)))

        return action