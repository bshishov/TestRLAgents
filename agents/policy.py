import numpy as np


class Policy(object):
    def get_action(self, state, available_actions):
        raise NotImplementedError


class QEGreedyPolicy(Policy):
    def __init__(self, q, epsilon=0.5, epsilon_decay=0.99, epsilon_min=0.1):
        self.epsilon = epsilon
        self.q = q
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_action(self, state, available_actions):
        probabilities = self.q.get_e_greedy_probabilites(state, self.epsilon)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return np.random.choice(available_actions, p=probabilities)


class QERandomPolicy(Policy):
    def __init__(self, q, epsilon=0.5, epsilon_decay=0.99, epsilon_min=0.1):
        self.epsilon = epsilon
        self.q = q
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_action(self, state, available_actions):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if np.random.rand() < self.epsilon:
            return available_actions[np.random.randint(len(available_actions))]
        return self.q.get_argmax_action(state)
