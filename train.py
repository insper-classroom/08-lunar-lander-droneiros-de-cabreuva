import csv
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
from DeepQLearning import DeepQLearning
from DeepQNetworks import DeepQNetworks
from hyperparameters import Hyperparameters

env = gym.make("LunarLander-v2")

print("Which model do you want to train?")
print("1 - DeepQLearning")
print("2 - DeepQNetworks")
model_id = int(input())

params = Hyperparameters(
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_dec=0.99,
    episodes=1_000,
    batch_size=64,
    memory_size=10_000,
)
memory = deque(maxlen=params.memory_size)

if model_id == 1:
    model = DeepQLearning(env, params, memory)
else:
    params.target_update_rate = 50
    model = DeepQNetworks(env, params, memory)

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
