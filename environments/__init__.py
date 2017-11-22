class Environment(object):
    last_reward = None
    total_reward = 0

    def __init__(self, agent):
        self.agent = agent

    def step(self, autorestart=True):
        action = self.agent.observe(self.get_observations(), self.last_reward, self.get_available_actions(), self.is_done())
        self.last_reward = self.do_action(action)
        self.total_reward += self.last_reward
        if autorestart and self.is_done():
            print(self.total_reward)
            self.total_reward = 0
            self.reset()

    def get_observations(self):
        raise NotImplementedError

    def get_available_actions(self):
        raise NotImplementedError

    def do_action(self, action):
        # TODO: Change state here
        raise NotImplementedError

    def is_done(self):
        return False

    def reset(self):
        raise NotImplementedError
