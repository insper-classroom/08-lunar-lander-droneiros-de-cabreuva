import torch


def create_network(env):
    model = torch.nn.Sequential(
        torch.nn.Linear(env.observation_space.shape[0], 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, env.action_space.n),
    )
    return model
