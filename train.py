import gym
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from collections import deque
from dqn_agent import DQN

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 5e-4
MEMORY_SIZE = 50000
BATCH_SIZE = 128
TARGET_UPDATE = 2000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.997
NUM_EPISODES = 1000
MAX_STEPS = 5000
STACK_SIZE = 4
FRAME_SKIP = 4  # Faster training

# Preprocessing
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64))  # Reduce size for speed
    return resized.astype(np.uint8)

def stack_frames(frames, new_frame):
    frames.append(new_frame)
    while len(frames) < STACK_SIZE:
        frames.append(new_frame)
    return np.stack(frames, axis=0)

def sample_memory(memory):
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        torch.tensor(np.array(states), dtype=torch.float32).to(device),
        torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device),
        torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
        torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
        torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment
env = gym.make("ALE/Pacman-v5", render_mode="rgb_array")
num_actions = env.action_space.n
frame = env.reset()[0]
processed = preprocess(frame)
state_stack = deque([processed] * STACK_SIZE, maxlen=STACK_SIZE)

# Networks
policy_net = DQN((STACK_SIZE, 64, 64), num_actions).to(device)
target_net = DQN((STACK_SIZE, 64, 64), num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# Memory
memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_START
step_count = 0
best_reward = -float("inf")

# Training Loop
for episode in range(NUM_EPISODES):
    frame = env.reset()[0]
    state_stack = deque([preprocess(frame)] * STACK_SIZE, maxlen=STACK_SIZE)
    state = np.stack(state_stack, axis=0)
    total_reward = 0

    for t in range(0, MAX_STEPS, FRAME_SKIP):
        # Action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        # Step with frame skip
        total_step_reward = 0
        for _ in range(FRAME_SKIP):
            next_frame, reward, terminated, truncated, _ = env.step(action)
            total_step_reward += reward
            if terminated or truncated:
                break

        done = terminated or truncated
        next_processed = preprocess(next_frame)
        next_state_stack = state_stack.copy()
        next_state_stack.append(next_processed)
        next_state = np.stack(next_state_stack, axis=0)

        memory.append((state, action, total_step_reward, next_state, done))
        state = next_state
        state_stack = next_state_stack
        total_reward += total_step_reward
        step_count += 1

        # Learn every few steps
        if step_count % 4 == 0 and len(memory) >= BATCH_SIZE:
            s, a, r, s_next, d = sample_memory(memory)

            # Double DQN: use policy_net to pick action, target_net to get value
            next_actions = policy_net(s_next).argmax(1, keepdim=True)
            max_next_q = target_net(s_next).gather(1, next_actions)

            target = r + GAMMA * max_next_q * (1 - d)
            q_values = policy_net(s).gather(1, a)

            loss = nn.SmoothL1Loss()(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target net
        if step_count % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    # Save best model
    if total_reward > best_reward:
        best_reward = total_reward
        best_model_path = "models/best_pacman_dqn.pth"
        torch.save(policy_net.state_dict(), best_model_path)

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"Episode {episode+1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    # Log reward
    with open("reward_log.csv", "a") as f:
        f.write(f"{episode+1},{total_reward:.2f},{epsilon:.3f}\n")

# Save final model
os.makedirs("models", exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_path = f"models/pacman_dqn_{timestamp}.pth"
torch.save(policy_net.state_dict(), model_path)
print(f"\n✅ Model saved to {model_path}")

env.close()