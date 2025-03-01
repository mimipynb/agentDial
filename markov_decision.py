"""

    markov_decision.py

    Define the trial runs as a Discrete Markov Decision-making problem.

    A Discrete Transition matrix is defined for selecting the action states: increasing, decreasing, no change

"""

import numpy as np
from handler import Trial

class DiscreteMarkov:
    def __init__(self, num_action, action_labels):
        """ E.g. action to increase / decrease / stay. """

        self.N = num_action
        self.labels = action_labels
        self.A = np.eye(self.N)
        self.prev_action = None

    def update(self, next_state, reward):
        """ Updates the transition matrix. """

        curr_idx = self.action_labels.index(self.prev_action)
        next_idx = self.action_labels.index(next_state)
        self.A[curr_idx, next_idx] += reward
        self.A /= self.A.sum(axis=1, keepdims=True)

    def next_action(self):
        """ Determines the next action based on the agents transition matrix (accumulated memory)"""

        if self.prev_action is not None:
            print("Proba scores: ", self.A[self.prev_action, :])
            return np.argmax(self.A[self.prev_action, :])
        else:
            return np.random.choice(self.A.shape[0])

    def __call__(self, curr_state):
        """ Takes in the ChatSession dataclass. higher_level is the parent transition matrix. """

        if curr_state[-1].role != 'assistant' or curr_state[-2].role != 'user':
            raise ValueError

        # 1. Collects feature from input state. NOTE: both feats ranges between 0.0 and 1.0.
        agent_emote_state = curr_state[-2]['inference']['agent_emote_state']
        dialog_emote_intensity = curr_state.data[['dialog_emotes']]

        # 2. [Temporary] Calculate the reward agent_emote_state (predicted from dialogue) vs. dialog_emote_intensity (how volatile the intensity expressed within the chat dialogue of both speakers - time series evaluation)
        agent_reward = np.max(agent_emote_state, np.mean(dialog_emote_intensity))

        # 3. Fetch the parameter's next action
        param_action = self.next_action()

        # 4. Updates the transition matrix.
        self.update(next_state=param_action, reward=agent_reward)

        return param_action

class Markov(Trial):
    def __init__(self):
        """ Main runner for a set of decoding params. Specifically: Temperature, top_p, and top_k. """

        super().__init__()
        self.action = {}
        for param in self.state_labels:
            self.action[param] = DiscreteMarkov(num_action=3, action_labels=self.action_labels)

    def run(self, state):
        """ Main and returns the Decoding Params. TODO: make this code more simply omg """

        for param, next_action in self.action.items():
            # fetches next action given state
            action = next_action(state)
            print(f"Param: {param}: {next_action}")
            # updates the rate of change
            self.state.adjust_meter(action=action, param=param)

        return {
            'temperature': self.state.temperature.state,
            'top_k': self.state.top_k.state,
            'top_p': self.state.top_p.state
        }
