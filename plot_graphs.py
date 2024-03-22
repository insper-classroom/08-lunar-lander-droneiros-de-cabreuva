import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

model_name = "dql"
# Load the data
df = pd.read_csv(f"results/ll_{model_name}_results.csv", header=None)
df.columns = ["episode", "reward"]

window_size = 20
df["rewards_mean"] = df["reward"].rolling(window=window_size).mean()
# Plot the data
sns.set_theme(style="darkgrid")
sns.lineplot(data=df, x="episode", y="rewards_mean")
plt.xlabel("Episodes")
plt.ylabel(f"Mean of Rewards (Window Size={window_size})")
# TODO: Colocar hyperparametros no titulo
plt.title("Rewards vs Episodes")
plt.savefig(f"results/ll_{model_name}_results.jpg", dpi=300)
