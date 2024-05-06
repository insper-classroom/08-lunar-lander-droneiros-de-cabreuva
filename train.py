import csv
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
from DeepQLearning import DeepQLearning
# from DeepQNetworks import DeepQNetworks
from DeepQNetworks import DQNAgent
from hyperparameters import Hyperparameters
import numpy as np

env = gym.make("LunarLander-v2")
np.random.seed(42)

print("Which model do you want to train?")
print("1 - DeepQLearning")
print("2 - DeepQNetworks")
model_id = int(input())

params = Hyperparameters(
    learning_rate=1e-4,
    gamma=0.99,
    epsilon=0.9,
    epsilon_min=0.05,
    epsilon_dec=1000,
    episodes=600,
    batch_size=128,
    memory_size=50_000,
)

if model_id == 1:
    memory = deque(maxlen=params.memory_size)   
    model = DeepQLearning(env, params, memory)
else:
    params.target_update_rate = 0.005
    model = DQNAgent(env, params)

device = model.device
print(f"Training model {model} using {device}")
rewards = model.train()

plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards vs Episodes")
plt.savefig(f"results/{model}_ll.jpg", dpi=300)
plt.close()

with open(
    f"results/ll_{model}_results.csv", "w+", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Reward"])
    for episode, reward in enumerate(rewards):
        writer.writerow([episode, reward])
