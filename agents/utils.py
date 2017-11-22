import math


def estimated_discounted_return(rewards, discount_factor=0.9):
    episode_length = len(rewards)
    estimated_return = []
    for t in range(episode_length):
        estimated_return.append(rewards[t])
        for t1 in range(t + 1, episode_length):
            estimated_return[t] += math.pow(discount_factor, t1 - t) * rewards[t1]
    return estimated_return
