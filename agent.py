

class Agent(object):
    def episode_start(self, state_space, action_space):
        pass

    def step(self, state, reward, available_actions):
        raise NotImplementedError

    def episode_end(self):
        pass