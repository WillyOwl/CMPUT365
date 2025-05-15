import numpy as np
import matplotlib.pyplot as plt

# Task 1: Setup
class Non_stationary_bandit:
    def __init__(self, k=10, random_walk_std=0.01):
        self.k = k
        self.q_star = np.zeros(k)
        self.random_walk_std = random_walk_std

    def step_random_walk(self, action):
        # Take random walk for all actions in terms of q_star
        self.q_star = np.random.normal(0, self.random_walk_std, self.k)

        """
        Choose the normal distribution with mean of q_star[action], and variance of 1
        according to the definition in the textbook
        """

        # Note that the second argument of np.random.normal stands for standard deviation
        # instead of variance
        return np.random.normal(self.q_star[action], 1)

# Task 2: Agent Implementations
class Sample_Average_Agent:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def choose_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.k - 1)

        else: return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

class Constant_Step_Size_Agent:
    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(k)

    def choose_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.k - 1)

        else: return np.argmax(self.Q)

    def update(self, action, reward):
        self.Q[action] += self.alpha * (reward - self.Q[action])

# Task 3: Experiment Execution

def run_experiment(agent_class, steps=10000, runs=100, **agent_args):
    rewards = np.zeros(steps)
    optimal_actions = np.zeros(steps)

    for run in range(runs):
        bandit = Non_stationary_bandit()
        agent = agent_class(**agent_args)

        for step in range(steps):
            action = agent.choose_action()
            reward = bandit.step_random_walk(action)
            agent.update(action, reward)

            rewards[step] += reward

            if action == np.argmax(bandit.q_star):
                optimal_actions[step] += 1

    return rewards / runs, optimal_actions / runs * 100

# Task 4: Plotting
def plot_results(sample_avg_rewards, sample_avg_optimal_percentage,
                 constant_step_rewards, constant_step_optimal_percentage):
    plt.figure(figsize=(12, 5))

    # Reward comparison
    plt.subplot(1, 2, 1)
    plt.plot(sample_avg_rewards, label='Sample Average')
    plt.plot(constant_step_rewards, label='Constant Step (α=0.1)')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    # Optimal action comparison
    plt.subplot(1, 2, 2)
    plt.plot(sample_avg_optimal_percentage, label='Sample Average')
    plt.plot(constant_step_optimal_percentage, label='Constant Step (α=0.1)')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sample_avg_rewards, sample_avg_optimal_percentage = run_experiment(Sample_Average_Agent, epsilon=0.1)

    constant_step_rewards, constant_step_optimal_percentage = run_experiment(Constant_Step_Size_Agent,
                                                                             epsilon=0.1, alpha=0.1)

    plot_results(sample_avg_rewards, sample_avg_optimal_percentage,
                 constant_step_rewards, constant_step_optimal_percentage)