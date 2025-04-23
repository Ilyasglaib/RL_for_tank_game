import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQNTank(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNTank, self).__init__()
        c, h, w = input_shape  # (3, 22, 22)

        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 

        # flatten size = 64 × 22 × 22
        self.fc1 = nn.Linear(64 * 22 * 22, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x / 255.0  # normalizing 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)
    
# A type of replay buffers that samples with bias towards transition with high TD error   
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0

        self.alpha = alpha  # how much prioritization
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = 1e-5

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = np.array(self.priorities)
        else:
            prios = np.array(self.priorities[:len(self.buffer)])

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance Sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            indices
        )

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = abs(td_error.item()) + self.epsilon

    def __len__(self):
        return len(self.buffer)


#The agentS
class DQNAgent:
    def __init__(self, input_shape=(12, 22, 22), num_actions=6, lr=1e-5, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=70000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.model = DQNTank(input_shape, num_actions)
        self.target_model = DQNTank(input_shape, num_actions)
        self.update_target()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        self.steps_done += 1
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                        np.exp(-1. * self.steps_done / self.epsilon_decay)

        if random.random() < eps_threshold:
            return random.randint(0, self.num_actions - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 21, 21)
            with torch.no_grad():
                q_values = self.model(state)
                return q_values.argmax().item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.permute(0, 3, 1, 2)  # (B, 3, 22, 22)
        next_states = next_states.permute(0, 3, 1, 2)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    #Another version of training but with the PER 
    def train_step_PER(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(batch_size)

        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)
        weights = weights.to(states.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q_values * (1 - dones)

        td_errors = q_values - expected_q.detach()
        loss = (td_errors ** 2 * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        #Clip gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities
        replay_buffer.update_priorities(indices, td_errors)
