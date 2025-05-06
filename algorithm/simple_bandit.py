import numpy as np
import random
import matplotlib.pyplot as plt

class Bandit:
    # Initialization
    def __init__(self, k):
        self.k = k # number of arms
        self.q_true = np.random.normal(0, 1, k) # true value of each arm from standard normal distribution
        self.q_est = np.zeros(k) # Estimated value of each arm
        self.N = np.zeros(k) # number of actions selected for each arm

    def bandit(self, action):
        return np.random.normal(self.q_true[action], 1)

    def select_action(self, epsilon):
        if random.random() < epsilon: # breaking policy is a randomly generated number between [0, 1) is less than epsilon
            return random.randint(0, self.k - 1) # closed interval

        else: return np.argmax(self.q_est)

    def run(self, epsilon, steps):
        rewards = np.zeros(steps)

        for step in range(steps):
            action = self.select_action(epsilon)

            reward = self.bandit(action)

            self.N[action] += 1
            self.q_est[action] += (1 / self.N[action]) * (reward - self.q_est[action])

            rewards[step] = reward

        return reward

if __name__ == "__main__":
    k = 10
    epsilon = 0.5
    steps = 2000

    bandit = Bandit(k)

    rewards = bandit.run(epsilon, steps)

    cumulative_rewards = np.cumsum(rewards)

    plt.plot(cumulative_rewards)

    plt.xlabel('steps')
    plt.ylabel('cumulative rewards')
    plt.title('Simple Bandit Algorithm')
    plt.show()
