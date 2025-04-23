import numpy as np
from base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def select_action(self, state):
        return self.action_space.sample()  # Sélectionne une action au hasard
    
    def learn(self, *args, **kwargs):
        pass