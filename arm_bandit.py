"""

    arm_bandit.py

    Define the trial runs as a Multi-arm bandit problem.

    Objective:
    - Maximize the cumulative reward over T trials for each arm.
    - How the agent can balance exploration and exploitation to maximize its cumulative reward for independent arms.

    Environment setup:
    - Reward:
        - adjust-param: 1 = increase, -1 = decrease, 0 = Do not change => sentimental inference of the utterance
        - select-param to adjust based on the
    - Arms: K independent arms, each with unknown reward distribution.

"""

from .handler import Trial
from typing import Literal

class MultiArm(Trial):
    def __init__(self, method: Literal['epsilon', 'thompson', 'UCB'] = 'epsilon'):
        self.method = method