# Agent Dial

Submodule of decoding parameter tuners that dynamically adjust to the agent’s cognitive state and linguistic analysis from current chat window.

## Usage

This operates with my other submodule `cognition` which returns collective features inferenced from the (i) the human user, and (ii) the chat conversation. The marginal probability density function of the agent’s cognitive states is computed by examining the conditional probability of the user's emotional trajectory over time (note: user considered in this submodule doesn't consider the activator (me) or any members with relations of higher priority than friend-zone e.g. stranger, friend.)

Dialogue conversations are analyzed using time-series methods, such as measuring the volatility of expression states. This operates as a lower and upper boundaries for the following chat windows. An example of this application is that if a user’s sentiment fluctuates significantly—shifting rapidly between frustration and neutrality—this configures decoding parameters according to the detected emotional instability, allowing the agent to return responses accordingly like offering reassurance.. etc.

In summary:

- Control of LLMs response generations conditioned to emotion states and unique to user (but not limited to the activator - me)
- Invoking chain of thought in LLMs
- Invoking second, triple, ... etc. responses

## Available Methods

To adapt the chat session, the following Reinforcement Learning methods were considered:

- [ ] Q-Learning problem
- [ ] Multi-arm bandit problem
- [x] Discrete Markov Decision Making
- [ ] Kalman Filtering

As usual, for the sake of simplicity, Kalman filtering will serve as the final method applied in my application code.

NOTE:

- The choice of trial depends on the agent’s environment setup. Treating this as a multi-armed bandit problem implies that the action states for each decoding parameter are independent of the others. Though, this assumption does not hold when applying Q-learning or Markov models.

## Dependencies

- `llm-cpp`
- `numpy`

## Base Environment setup

Describing the problem as a Reinforcement Learning problem:

- **Reward**: The rewards are based on the suggested emotional intensity to be expressed in the agent's response. This is *continous* reward states but can be treated as a discrete.
  - Emotions expressed from the user's utterance -> Lower Level Reward
  - Kernel Density of the Emotional intensity over the chat dialogue -> Higher Level Reward
- **Observation**:
  - Interaction between human and bot (`ChatSession` from `basement` repo)
- **Agent's Action**:
  - Decision-making involves increasing, decreasing, or maintaining the current parameter.
  - Decision-making also determines whether to retain the current parameter or switch to another.

## RL Brief Background

Common strategies used for this type of RL problem:

- ε-Greedy: Selects the best-known action with probability (1 - ε) and explores randomly with probability ε.
- Upper Confidence Bound (UCB): Selects actions by considering their potential for improvement.
- Thompson Sampling: Uses Bayesian inference to estimate reward distributions.
