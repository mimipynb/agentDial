"""

    q_learning.py

    Define the trial runs as a Q-learning RL problem.

    Q-Learning Setup:
    - Q-table:
        - A table of state-action values updated using the Bellman equation.
        - Stores learned Q-values for each (state, action) pair.
    - Learning Rate (α):
        - Controls how much newly acquired information overrides old information.
    - Discount Factor (γ):
        - Determines the importance of future rewards.
    - Exploration-Exploitation:
        - ε-greedy policy used to balance exploration (trying new actions) and exploitation (choosing known best actions).

"""

import numpy as np
from typing import Literal
from .handler import Trial

class QLearner(Trial):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Q-learning agent for adjusting decoding parameters.

        :param alpha: Learning rate
        :param gamma: Discount factor
        :param epsilon: Exploration rate for epsilon-greedy policy
        :numpy Q: calculates the accumulated reward for taking a specific action given current state.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((self.N, self.N)) # Q-matrix

