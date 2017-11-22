from collections import deque
import random
import numpy as np

from agents import Agent


class DeepQAgent(Agent):
    def __init__(self, q, batch_size=32, discount=0.95, memory_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_memory = deque(maxlen=memory_size)
        self.discount = discount
        self.batch_size = batch_size
        self.q = q
        self.last_state = None
        self.last_action = None

    def step(self, state, reward, available_actions, done):
        if self.last_action is not None and self.last_state is not None:
            self.replay_memory.append([self.last_state, self.last_action, reward, state, int(done)])

        if len(self.replay_memory) > self.batch_size:
            self.train_on_replay_memory()

        action = self.policy.get_action(state, available_actions)

        self.last_action = action
        self.last_state = state

        return action

    def train_on_replay_memory(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        batch = np.array(batch)

        states = np.array([x for x in batch[:, 0]])
        actions = np.array([x for x in batch[:, 1]])
        rewards = np.array([x for x in batch[:, 2]])
        new_states = np.array([x for x in batch[:, 3]])

        # DQN Targets:
        # Target = R(t+1) + gamma * max[a] Q(S(t+1), a, w)
        target_rewards = rewards + self.discount * np.max(self.q.get_batch(new_states), axis=1)

        # Set non-discounted rewards for terminal states
        target_rewards[batch[:, 4] == 1] = rewards[batch[:, 4] == 1]
        current_q = self.q.get_batch(states)
        target_q = current_q.copy()
        for i in range(len(current_q)):
            target_q[i][actions[i]] = target_rewards[i]

        self.q.set_batch(states, target_q)


class DeepQAgentPlain(Agent):
    DISCOUNT_FACTOR = 0.95
    EPSILON = 0.5
    EPSILON_DECAY = 0.999
    EPSILON_MIN = 0.1
    REPLAY_MEMORY_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 1

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.model = model
        self.last_state = None
        self.step_n = 0
        self.epsilon = self.EPSILON
        self.last_action = None

    def step(self, state, reward, available_actions, done):
        if self.last_action is not None and self.last_state is not None:
            self.replay_memory.append([self.last_state, self.last_action, reward, state, int(done)])

        if len(self.replay_memory) > self.BATCH_SIZE:
            self.train_on_replay_memory()

        self.step_n += 1

        # e-greedy exploration
        if np.random.rand() < self.epsilon or state is None:
            action_idx = random.randint(0, len(available_actions) - 1)
            action = available_actions[action_idx]
        else:
            action = np.argmax(self.model.predict(state))

        self.last_action = action
        self.last_state = state

        self.epsilon = max(self.epsilon * self.EPSILON_DECAY, self.EPSILON_MIN)

        return action

    def train_on_replay_memory(self):
        for epoch in range(self.EPOCHS):
            batch = random.sample(self.replay_memory, self.BATCH_SIZE)
            batch = np.array(batch)

            states = np.array([x for x in batch[:, 0]])
            actions = np.array([x for x in batch[:, 1]])
            rewards = np.array([x for x in batch[:, 2]])
            new_states = np.array([x for x in batch[:, 3]])

            target_rewards = rewards + self.DISCOUNT_FACTOR * np.max(self.model.predict_batch(new_states), axis=1)
            target_rewards[batch[:, 4] == 1] = rewards[batch[:, 4] == 1]
            current_q = self.model.predict_batch(states)
            target_q = current_q.copy()
            for i in range(len(current_q)):
                target_q[i][actions[i]] = target_rewards[i]

            self.model.fit_batch(states, target_q)


class DoubleDeepQAgent(Agent):
    def __init__(self, q1, q2, batch_size=32, discount=0.95, memory_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_memory = deque(maxlen=memory_size)
        self.discount = discount
        self.batch_size = batch_size
        self.q1 = q1
        self.q2 = q2
        self.last_state = None
        self.last_action = None

        # Use the first q as policy
        self.policy.q = self.q1

    def step(self, state, reward, available_actions, done):
        if self.last_action is not None and self.last_state is not None:
            self.replay_memory.append([self.last_state, self.last_action, reward, state, int(done)])

        if len(self.replay_memory) > self.batch_size:
            self.train_on_replay_memory()

        action = self.policy.get_action(state, available_actions)

        self.last_action = action
        self.last_state = state

        return action

    def train_on_replay_memory(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        batch = np.array(batch)

        states = np.array([x for x in batch[:, 0]])
        actions = np.array([x for x in batch[:, 1]])
        rewards = np.array([x for x in batch[:, 2]])
        new_states = np.array([x for x in batch[:, 3]])

        target_rewards = np.zeros(len(states))
        i = 0

        # Double DQN:
        # Target = R(t+1) + gamma * Q(   S(t+1),    argmax[a] (Q(S(t+1), a, w),   w1)
        # TODO: IMPLEMENT EFFICIENTLY
        for state, action, reward, new_state, _ in batch:
            target_rewards[i] = reward + self.discount * self.q2[new_state, self.q1.get_argmax_action(new_state)]
            i += 1

        # Swap networks
        self.q1, self.q2 = self.q2, self.q1

        # Use the first q as policy
        self.policy.q = self.q1

        # Set non-discounted rewards for terminal states
        target_rewards[batch[:, 4] == 1] = rewards[batch[:, 4] == 1]
        current_q = self.q1.get_batch(states)
        target_q = current_q.copy()
        for i in range(len(current_q)):
            target_q[i][actions[i]] = target_rewards[i]

        self.q1.set_batch(states, target_q)


class DuelingDeepQAgent(Agent):
    def step(self, state, reward, available_actions, done):
        pass

    # TODO: IMPLEMENT
