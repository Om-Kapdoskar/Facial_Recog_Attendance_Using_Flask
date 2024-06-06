import gym
import numpy as np

# Create the environment
env = gym.make("FrozenLake-v1")

# Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Define Q-learning parameters
gamma = 0.9  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Exploration rate

# Define helper function to choose action using epsilon-greedy policy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Random action
    else:
        return np.argmax(q_table[state])

# Q-learning algorithm
num_episodes = 1000
for _ in range(num_episodes):
    state = env.reset()  # Reset environment to initial state
    done = False  # Flag to indicate if episode is finished
    while not done:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        # Convert state to integer index
        state_index = env.encode(*state)
        next_state_index = env.encode(*next_state)
        # Q-learning update rule
        q_table[state_index, action] += alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action])
        state = next_state

# Test the trained agent
state = env.reset()
done = False
while not done:
    # Convert state to integer index
    state_index = env.encode(*state)
    action = np.argmax(q_table[state_index])
    next_state, reward, done, _ = env.step(action)
    env.render()  # Display current state
    state = next_state
