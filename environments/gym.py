from environments import Environment
import gym
import gym.spaces


class GymEnvironment(Environment):
    state = None
    done = False
    available_actions = None

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.available_actions = [x for x in range(self.env.action_space.n)]

        self.state = self.env.reset()

    def do_action(self, action):
        new_state, reward, done, _ = self.env.step(action)
        self.state = new_state
        self.done = done
        return reward

    def get_available_actions(self):
        return self.available_actions

    def reset(self):
        return self.env.reset()

    def is_done(self):
        return self.done

    def render(self):
        return self.env.render()

    def get_observations(self):
        return self.state
