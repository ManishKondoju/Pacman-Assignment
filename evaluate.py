import gym
import torch
import cv2
import numpy as np
from collections import deque
from dqn_agent import DQN
import os

# Constants
STACK_SIZE = 4
FRAME_SKIP = 4
MODEL_PATH = "models/best_pacman_dqn.pth"

# Preprocess frame
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64))
    return resized.astype(np.uint8)

# Stack frames
def stack_frames(frames, new_frame):
    frames.append(new_frame)
    while len(frames) < STACK_SIZE:
        frames.append(new_frame)
    return np.stack(frames, axis=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup environment
env = gym.make("ALE/Pacman-v5", render_mode="human")
num_actions = env.action_space.n

# Load model
policy_net = DQN((STACK_SIZE, 64, 64), num_actions).to(device)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy_net.eval()
print(f"📦 Loaded model: {MODEL_PATH}")

# Reset env
frame = env.reset()[0]
state_stack = deque([preprocess(frame)] * STACK_SIZE, maxlen=STACK_SIZE)
state = np.stack(state_stack, axis=0)
total_reward = 0

# Run one episode
done = False
step_count = 0
while not done:
    with torch.no_grad():
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        q_values = policy_net(state_tensor)
        action = q_values.argmax().item()

    total_step_reward = 0
    for _ in range(FRAME_SKIP):
        next_frame, reward, terminated, truncated, _ = env.step(action)
        total_step_reward += reward
        done = terminated or truncated
        if done:
            break

    next_processed = preprocess(next_frame)
    state_stack.append(next_processed)
    state = np.stack(state_stack, axis=0)
    total_reward += total_step_reward
    step_count += 1

    if step_count % 20 == 0:
        print(f"🕹️ Step {step_count} | Total Reward: {total_reward:.2f}")

env.close()
print(f"\n✅ Evaluation completed! Total reward: {total_reward:.2f}")
