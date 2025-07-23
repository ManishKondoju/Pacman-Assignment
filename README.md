# 🧠 Pacman DQN Agent 🎮

This project implements a **Deep Q-Learning (DQN)** agent to play **Atari Pacman** using PyTorch and OpenAI Gym (ALE). The goal is to train a neural network to maximize cumulative reward by learning optimal actions in the Pacman environment.

---

---

## 🛠️ Setup Instructions

> ❗ Do NOT include your virtual environment in the repo. Create one fresh.

### 1. Create a Python environment

```bash
python3 -m venv pacman_dqn_env
source pacman_dqn_env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, install manually:

```bash
pip install gymnasium[atari,accept-rom-license]
pip install torch torchvision opencv-python numpy matplotlib
```

---

## 🚀 Training

Run the training script with default hyperparameters:

```bash
python train.py
```

- Trains for 1000 episodes
- Saves the best model to `models/best_pacman_dqn.pth`
- Logs reward per episode

---

## 🧪 Evaluation

Evaluate the trained agent:

```bash
python evaluate.py
```

- Renders Pacman playing live
- Outputs total reward and actions taken
- Uses stacked frames and frame skipping for smoother play

---

## 📊 Hyperparameters (Optimized)

| Parameter        | Value    |
|------------------|----------|
| GAMMA            | 0.99     |
| LEARNING_RATE    | 5e-4     |
| BATCH_SIZE       | 128      |
| MEMORY_SIZE      | 50,000   |
| EPSILON_START    | 1.0      |
| EPSILON_END      | 0.01     |
| EPSILON_DECAY    | 0.997    |
| TARGET_UPDATE    | 2000     |
| STACK_SIZE       | 4        |
| FRAME_SKIP       | 4        |
| MAX_STEPS        | 5000     |
| NUM_EPISODES     | 1000     |

---

## 🎮 Pacman Scoring System

| Action                 | Reward |
|------------------------|--------|
| Eating Pellet          | +10    |
| Eating Ghost (after power pill) | +200 |
| Eating Fruit           | +100   |
| Getting Caught by Ghost | -500  |
| Game Over              | -1000  |

---

## 🧠 Architecture

- **CNN-based Dueling DQN**
- Convolution layers extract features
- Two separate streams for **Advantage** and **Value**
- Combines using:  
  \[
  Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a')\right)
  \]

---

## 📝 License & Credits

- ROM and environment via [Arcade Learning Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment)
- Deep Q-Learning concepts adapted from [DeepMind’s DQN paper (2015)](https://www.nature.com/articles/nature14236)
- Inspired by OpenAI Gymnasium & PyTorch

---

## 📌 Author

**Manish Kumar Kondoju**  
M.S. in Information Systems – Northeastern University  
[GitHub Profile](https://github.com/ManishKondoju)
