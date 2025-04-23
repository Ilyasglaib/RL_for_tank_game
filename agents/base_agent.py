class BaseAgent:
    """Classe de base pour les agents RL."""
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def select_action(self, state):
        raise NotImplementedError

    def learn(self, *args, **kwargs):
        raise NotImplementedError