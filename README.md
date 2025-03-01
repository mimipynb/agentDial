# Agent Dial

CLI tool containing decoding parameters tuners that adapt to the agent's current cognitive state and user interaction.

## Available Methods

To adapt the chat session, the following Reinforcement Learning methods are considered:

- [ ] Q-Learning problem
- [ ] Multi-arm bandit problem
- [ ] Discrete Markov Decision Making

NOTE:

- The choice of trial depends on the agent's environment setup. If the action states (decoding parameters) are altered independently then Multi-arm bandit problem and be used. Otherwise, Q-Learning / Markov is applied.

## Dependencies

- `basement`: from my gitlab repo
- `agenthub`: from my gitlab repo

## Base Environment setup

Describing the problem as a Reinforcement Learning problem:

- **Reward**: The rewards are based on the suggested emotional intensity to be expressed in the agent's response. This is *continous* reward states but can be treated as a discrete. 
  - Emotions expressed from the user's utterance -> Lower Level Reward
  - Kernel Density of the Emotional intensity over the chat dialogue -> Higher Level Reward
- **Observation**:
  - User's chat message (`ChatMessage`)
- **Agent's Action**:
    *Assuming no run asynchronously*
  - Decision making in increasing, decreasing or no change keep this parameter still.
  - Decision making in staying or switching to another parameter.

## RL Brief Background

Common strategies used for this type of RL problem:

- ε-Greedy: Selects the best-known action with probability (1 - ε) and explores randomly with probability ε.
- Softmax Action Selection: Uses a probabilistic approach to choose actions based on learned Q-values.
- Upper Confidence Bound (UCB): Selects actions by considering their potential for improvement.
- Thompson Sampling: Uses Bayesian inference to estimate reward distributions.
