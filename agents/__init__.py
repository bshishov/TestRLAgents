class Processor(object):
    def to_state(self, observation):
        raise NotImplementedError


class SimpleProcessor(Processor):
    def to_state(self, observation):
        return observation


class Agent(object):
    def __init__(self, processor=SimpleProcessor(), policy=None):
        self.processor = processor
        self.policy = policy

    def observe(self, observations, reward, available_actions, done):
        return self.step(self.processor.to_state(observations), reward, available_actions, done)

    def step(self, state, reward, available_actions, done):
        raise NotImplementedError
