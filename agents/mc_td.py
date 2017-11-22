from agents import Agent
from agents.utils import estimated_discounted_return
from agents.fn import Counter, TableValueFunction, TableQFunction


class MonteCarloAgent(Agent):
    # The agent that learns from the full episode (so it waits until the end of the episode)
    # backing up the full estimate return for the episode
    # then updating Q and value function

    discount_factor = 0.9

    def __init__(self, q=TableQFunction(), v=TableValueFunction(), lr=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v = v  # value for given state
        self.q = q  # Action-value function
        self.lr = lr
        if lr is None:
            self.states_visited = Counter()  # number of states visited
        self.episode_history = []
        self.last_state = None
        self.last_action = None

    def step(self, state, reward, available_actions, done):
        if self.last_state is not None and self.last_action is not None:
            self.episode_history.append((self.last_state, self.last_action, reward))

        if done:
            self.learn_from_episode()

        action = self.policy.get_action(state, available_actions)

        self.last_state = state
        self.last_action = action

        if done:
            self.last_state = None
            self.last_action = None

        return action

    def learn_from_episode(self):
        # Step 1: Policy evaluation
        rewards = []
        states = []

        for state, action, reward in self.episode_history:
            rewards.append(reward)
            states.append(state)

        expected_return = estimated_discounted_return(rewards)
        t = 0
        for state, action, reward in self.episode_history:
            if self.lr is None:
                self.states_visited[state] += 1
                alpha = 1 / self.states_visited[state]
            else:
                alpha = self.lr

            v = self.v[state]
            v += alpha * (expected_return[t] - v)
            self.v[state] = v
            t += 1

            # Step 2: policy update:
            self.q[state, action] = v
        self.episode_history = []


class TemporalDifferenceAgent(Agent):
    def __init__(self, q=TableQFunction(), v=TableValueFunction(), lr=0.1, l=0, discount=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v = v
        self.q = q
        self.l = l  # lambda
        self.lr = lr  # learning rate
        self.discount = discount
        self.last_state = None
        self.last_action = None

    def step(self, state, reward, available_actions, done):
        if self.last_state is not None:
            # Update last state value with new information
            last_v = self.v[self.last_state]
            last_v += self.lr * (reward + self.discount * self.v[state] - last_v)
            self.v[self.last_state] = last_v

            # Update Q function for last action
            self.q[self.last_state, self.last_action] = last_v

        action = self.policy.get_action(state, available_actions)

        self.last_state = state
        self.last_action = action

        if done:
            self.last_state = None
            self.last_action = None

        return action


class Sarsa(Agent):
    def __init__(self, q=TableQFunction(), lr=0.1, discount=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q
        self.lr = lr  # learning rate
        self.discount = discount
        self.last_state = None
        self.last_action = None

    def step(self, state, reward, available_actions, done):
        action = self.policy.get_action(state, available_actions)

        if self.last_state is not None and self.last_action is not None:
            last_q = self.q[self.last_state, self.last_action]
            last_q += self.lr * (reward + self.discount * self.q[state, action] - last_q)
            self.q[self.last_state, self.last_action] = last_q

        self.last_state = state
        self.last_action = action

        if done:
            self.last_state = None
            self.last_action = None

        return action
