"""dqn network for final stage of conveyor belt segregation and extraction."""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from threshold_detections import run_yolo

# DQN Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64

# DQN Model Definition
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize DQN components
def create_dqn(input_dim, output_dim):
    return DQNetwork(input_dim, output_dim)

# Experience replay memory
memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# Action space: Accept (1) or Reject (0)
ACTION_SPACE = [0, 1]  # 0: Reject, 1: Accept

# Choosing an action using epsilon-greedy policy
def choose_action(state, epsilon, dqn):
    if np.random.rand() < epsilon:
        return random.choice(ACTION_SPACE)  # Exploration
    state = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        q_values = dqn(state)
    return torch.argmax(q_values).item()  # Exploitation

# Training the DQN
def train_dqn(dqn, target_dqn, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    
    batch = random.sample(memory, BATCH_SIZE)
    
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    
    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_q_values = target_dqn(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
    
    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Updating the target DQN
def update_target_dqn(dqn, target_dqn):
    target_dqn.load_state_dict(dqn.state_dict())


CONFIDENCE_THRESHOLD = 0.5  # Threshold for low-confidence items

# Define environment state
def get_state_from_detection(item):
    # Example state: [confidence_score]
    return [item['confidence']]

# Main function to integrate YOLO and DQN
def process_waste(image_path, dqn, target_dqn, epsilon, optimizer):
    detected_items = run_yolo(image_path)
    total_reward = 0

    for item in detected_items:
        confidence = item['confidence']

        # Skip if the confidence score is too low
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Get the state (confidence score in this case)
        state = get_state_from_detection(item)

        # Choose action (0: Reject, 1: Accept) using DQN
        action = choose_action(state, epsilon, dqn)

        # For simplicity, reward is based on correct classification (dummy rewards for now)
        reward = 1 if action == 1 else 0  # Accept or reject based on action
        
        # Next state (if needed)
        next_state = get_state_from_detection(item)

        # Store experience in replay memory
        memory.append((state, action, reward, next_state, False))

        # Train DQN
        train_dqn(dqn, target_dqn, optimizer)

        total_reward += reward

    return total_reward


# Initialize DQN and target DQN
input_dim = 1  # Example: confidence score as the input state
output_dim = len(ACTION_SPACE)
dqn = create_dqn(input_dim, output_dim)
target_dqn = create_dqn(input_dim, output_dim)
optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)

# Training loop
for episode in range(1000):  # Number of training episodes
    image_path = 'path_to_image.jpg'  # Example image path
    total_reward = process_waste(image_path, dqn, target_dqn, EPSILON, optimizer)
    
    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY
    
    # Periodically update the target DQN
    if episode % 10 == 0:
        update_target_dqn(dqn, target_dqn)
    
    print(f"Episode {episode}, Total Reward: {total_reward}")
