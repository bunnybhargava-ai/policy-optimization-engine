# policy-optimization-engine
# Policy Iteration for Dynamic Policy Optimization
A simplified version of the policy iteration algorithm applied to discrete Markov Decision Processes. This repository demonstrates exact value function convergence and policy stabilization using iterative dynamic programming techniques.

## Algorithmic Framework and Mathematics
For the current policy, the agent computes an optimal policy by alternating between decoupled phases inside a terminal feedback loop until the policy becomes completely stable.

### 1. Policy Evaluation
For the current policy, the agent computes the state-value function by iteratively solving the Bellman equation for all states drops below our defined limit.

### 2. Policy Improvement 
Once the value function stabilises, the agent updates its policy greedily by looking ahead at the expected values of all possible actions. If the newly generated policy matches the previous one across all the states, the policy is flagged as 'Stable', the loop terminates, and optimal behaviour is achieved.

## About me
Class 11 Student (PCM/CS) based in India, learning about algorithmic risk  management and mode-free reinforcement learning frameworks.
