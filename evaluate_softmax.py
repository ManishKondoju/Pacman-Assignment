import gym
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from dqn_agent import DQN
import time

# ------------------------------
# ✅ Add preprocessing functions directly
# ------------------------------
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64))
    return resized.astype(np.uint8)

def stack_frames(frames, new_frame, stack_size=4):
    frames.append(new_frame)
    while len(frames) < stack_size:
        frames.append(new_frame)
    return np.stack(frames, axis=0)

# ------------------------------
# 🔧 Setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("ALE/Pacman-v5", render_mode="human")
num_actions = env.action_space.n
stack_size = 4
input_shape = (stack_size, 64, 64)
TEMPERATURE = 1.0  # Try 0.5, 1.0, or 2.0 for different exploration

# Load model
policy_net = DQN(input_shape, num_actions).to(device)
policy_net.load_state_dict(torch.load("models/best_pacman_dqn.pth", map_location=device))
policy_net.eval()

# ------------------------------
# 🎮 Evaluate
# ------------------------------
num_episodes = 1
max_steps = 1000

for episode in range(num_episodes):
    obs, _ = env.reset()
    frame = preprocess_frame(obs)
    state_stack = [frame] * stack_size
    state = np.stack(state_stack, axis=0)
    total_reward = 0

    for step in range(max_steps):
        state_tensor = torch.tensor([state], dtype=torch.float32).to(device)

        with torch.no_grad():
            q_values = policy_net(state_tensor).squeeze()
            probs = F.softmax(q_values / TEMPERATURE, dim=0).cpu().numpy()
            action = np.random.choice(num_actions, p=probs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        next_frame = preprocess_frame(next_obs)
        state_stack.pop(0)
        state_stack.append(next_frame)
        state = np.stack(state_stack, axis=0)

        if step % 20 == 0:
            print(f"🕹️ Step {step} | Total Reward: {total_reward:.2f}")

        if done:
            break

    print(f"\n✅ Evaluation completed using Softmax Policy! Total reward: {total_reward:.2f}")

env.close()
