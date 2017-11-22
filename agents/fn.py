import numpy as np


class ValueFunction(object):
    def __init__(self):
        pass

    def get(self, state):
        raise NotImplementedError

    def set(self, state, value):
        raise NotImplementedError

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.set(key, value)


class TableValueFunction(ValueFunction):
    table = {}

    def get(self, state):
        return self.table[state]

    def set(self, state, value):
        self.table[state] = value


class ApproximateValueFunction(ValueFunction):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get(self, state):
        return self.model.predict(state)

    def set(self, state, value):
        self.model.fit(state, value)


class QFunction(object):
    def get(self, state):
        raise NotImplementedError

    def get_value(self, state, action):
        raise NotImplementedError

    def set(self, state, action, value):
        raise NotImplementedError

    def get_argmax_action(self, state):
        raise NotImplementedError

    def get_e_greedy_probabilites(self, state, epsilon):
        # TODO: IMPLEMENT
        raise NotImplementedError
        q_state = self.get(state)
        m = len(q_state)
        best_action = self.get_argmax_action(state)
        probabilites = []

        for action in q_state:
            if action == best_action:
                return epsilon / m + 1 - epsilon
            return epsilon / m

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.get_value(*item)
        return self.get(item)

    def __setitem__(self, key, value):
        self.set(*key, value)


class TableQFunction(QFunction):
    def __init__(self, default_v=0):
        self.q = {}
        self.default_v = default_v

    def get_argmax_action(self, state, default=None):
        if state not in self.q:
            return default
        qstate = self.q[state]
        return max(qstate, key=qstate.get)

    def set(self, state, action, value):
        if state not in self.q:
            self.q[state] = {action: value}
        else:
            q = self.q[state]
            q[action] = value

    def get_value(self, state, action):
        if state in self.q:
            return self.q[state].get(action, self.default_v)
        return self.default_v

    def get(self, state):
        return self.q.get(state, {})


class ArrayQFunction(QFunction):
    def __init__(self, states, actions, default_v=0):
        self.q = np.full((states, actions), default_v)

    def get_argmax_action(self, state):
        return np.argmax(self.q[state])

    def set(self, state, action, value):
        self.q[state, action] = value

    def get_value(self, state, action):
        return self.q[state, action]

    def get(self, state):
        return self.q[state]

    def __getitem__(self, *args, **kwargs):
        return self.q.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.q.__setitem__(*args, **kwargs)


class ApproximateQFunction(QFunction):
    def __init__(self, model):
        self.model = model

    def get_value(self, state, action):
        return self.model.predict(state)[action]

    def get(self, state):
        return self.model.predict(state)

    def get_batch(self, states):
        return self.model.predict_batch(states)

    def set(self, state, action, value):
        q = self.get(state)
        q[action] = value
        self.model.fit(state, q)

    def set_batch(self, states, q_values):
        self.model.fit_batch(states, q_values)

    def get_e_greedy_probabilites(self, state, epsilon):
        q_state = self.get(state)
        m = len(q_state)
        probs = np.zeros(len(q_state))
        probs.fill(epsilon / m)
        probs[np.argmax(q_state)] = epsilon / m + 1 - epsilon
        return probs

    def get_argmax_action(self, state):
        return np.argmax(self.model.predict(state))


class Counter(object):
    counts = {}

    def __init__(self, default=0):
        self.default_n = default

    def __getitem__(self, item):
        if item in self.counts:
            return self.counts[item]
        return self.default_n

    def __setitem__(self, key, value):
        self.counts[key] = value


if __name__ == '__main__':
    q = TableQFunction()
    q['abc', 1] = 0
    q['abc', 2] = 1
    q['abc', 3] = 2
    q['abc', 4] = 3
    #q['abc',:] = [1,2,3,4]
    print(q['abc'])
    print(q.get_argmax_action('abc'))
    print(q.get_argmax_action('aasd'))
    print(q.get_argmax_action('aasd', default=1))


    q = ArrayQFunction(10, 4)
    print(q[2:5, 3])
    print(q.get_argmax_action(2))