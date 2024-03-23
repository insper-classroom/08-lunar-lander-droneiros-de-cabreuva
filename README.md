# Lunar Lander Project

**Team:** Droneiros de Cabreúva
**Members:** Jorás, Pedro, Renato & Ricardo

## Overview
The Lunar Lander environment, as detailed on [Farama Foundation Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/), presents a simulation to work on a sollution to the classic rocket trajectory optimization problem, where the goal is to safely land a spacecraft on the lunar surface. The challenge involves balancing the lander using its thrusters, aiming for a smooth touchdown at a designated landing site.

![Lunar Lander Example Gif](https://gymnasium.farama.org/_images/lunar_lander.gif)

### Problem Description
In this environment, an agent controls the lander by making continuous adjustments to its position and velocity. The objective is to land the craft between two flags on the surface with minimal fuel usage, avoiding crashes and hard landings. Successful landings and specific maneuvers, like coming to a rest near the landing zone, are rewarded, while excessive fuel usage and crashing are penalized.

### Action Space
The action space consists of four discrete actions:
- 0: Do nothing
- 1: Fire the left orientation engine
- 2: Fire the main engine
- 3: Fire the right orientation engine

These actions allow the agent to control the lander's lateral movements and vertical thrust.

### Observation Space
The observation space is an 8-dimensional vector representing various aspects of the lander's state, including horizontal ($x$) and vertical ($y$) position, horizontal and vertical linear velocity, lander angle and angular velocity, and two booleans indicating whether each leg is in contact with the ground.

## Comparison: Deep Q-Learning vs DQN (Deep Q-Networks)

In this project, we have implemented both Deep Q-Learning and DQN (Deep Q-Networks) algorithms to address the challenge of landing a spacecraft in the Lunar Lander environment. Through the application of Deep Q-Learning, we explored the fundamental approach of utilizing deep neural networks to approximate the Q-value function, allowing the agent to predict the value of its actions in various states. This method laid the groundwork for our further exploration with the DQN algorithm, which builds upon Deep Q-Learning by introducing critical innovations such as experience replay and fixed Q-targets.

By comparing these two algorithms, we not only gauged the impact of advanced techniques like experience replay on the learning efficiency and stability but also demonstrated the evolution of reinforcement learning strategies from basic Q-Learning to sophisticated architectures like DQN. The implementation of both algorithms provided valuable insights into the dynamics of reinforcement learning and its application in complex environments such as lunar landing.

### Learning Curves

![Learning Curve Deep Q-Learning](results/ll_dql_results.jpg)

![Learning Curve DQN](results/dqn_ll.jpg)

The learning curve above demonstrates the agent's performance over time, measured in terms of average reward per episode. Initially, the agent struggles to achieve successful landings, often incurring penalties for crashes or excessive fuel consumption. Over time, as the agent learns from its experiences, we observe a positive trend in performance, with increased rewards indicating more successful and efficient landings.

### Agent Demonstrations

Here are two animations showing the agent trained using Deep Q-Learning algorithm in action:

- Initial Stages of Learning:
  
  ![Initial Stages of Deep Q-Learning](results/lander_initial_dql.pt.gif)

- After Training Completion:
  
  ![Trained Deep Q-Learning Agent](results/lander_trained_dql.pt.gif)

Here are two animations showing the agent trained using DQN algorithm in action:

- Initial Stages of Learning:
  
  ![Initial Stages of Deep Q-Networks](results/lander_initial_dqn.pt.gif)

- After Training Completion:
  
  ![Trained Deep Q-Networks Agent](results/lander_trained_dqn.pt.gif)

These GIFs illustrate the progression from an inexperienced agent to a skilled one, capable of handling the complexities of lunar landing.

### Conclusion

Conclusion

## References

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Khj4RN1-)
