# INF581-PROJECT-TANK

## Reinforcement Learning for Tank Game

This project presents a Reinforcement Learning (RL) approach to a custom tank game environment. Two learning strategies have been implemented and compared:

- A **Q-table**-based agent (tabular RL)
- A **Deep Q-Network (DQN)** agent (deep reinforcement learning)

## How to Play

To run the game, simply execute the following script:

```bash
python game_play.py
```

## Project Structure

### `agent.py`

This file contains the implementation of the agent classes used in the project. It includes:

- The neural network architecture for the DQN agent
- The training algorithm
- The replay buffer for experience sampling and training

> ⚠️ Note: `BaseAgent` and `RandomAgent` classes are present in the file but were **not used** in the final version of the project.

### `envs/` folder

This folder contains the full implementation of the game environment.

- **`game_elements.py`**: Contains the core logic of the game, including:
  - The `Tank` and `Projectile` classes
  - Actions and behaviors associated with these classes

- **`tank_env.py`**: Contains the RL-specific interface for the environment:
  - `reset()` function to initialize a new game state
  - `step(action)` function to process an action and return the new state, reward, and termination status

### `notebooks/` folder

This folder contains the implementation of the notebooks to run the training loop and the evaluation of the agents

- **`checkpoints`**: Contains two saved model that have been trained:
  -model_900: The neural network takes only 1 frame, no prioritized replay buffer,lr=10-4
  -model_2000: The neural network takes 4 frames, Prioritized replay buffer, lr=10-5 

- **`Evaluation.ipynb`**: Run to see the trained DQN agent (model_2000) playing in evaluation mode (without learning or epsilon greedy)

- **`DQN_algorithm.ipynb`** and **`Q_table_algorithm.ipynb`** : Training loops and plots of the moving average reward and moving average kill streak