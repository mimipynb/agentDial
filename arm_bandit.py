"""

    arm_bandit.py

    Define the trial runs as a Multi-arm bandit problem.

    Objective:
    - Maximize the cumulative reward over T trials for each arm.

"""

from .handler import Trial
from typing import Literal

class MultiArm(Trial):
    def __init__(self, method: Literal['epsilon', 'thompson', 'UCB'] = 'epsilon'):
        self.method = method