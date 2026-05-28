!pip install --user --upgrade gym==0.20.0

import numpy as np
import pprint
from gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)

env = GridworldEnv()
'''Policy Iteartion Program Part-1'''

def get_env_size(env):
    nS = env.nS if hasattr(env, "nS") else env.observation_space.n
    nA = env.nA if hasattr(env, "nA") else env.action_space.n
    return nS, nA


def action_estimation(env, state, V, discount_factor=1.0):
    """
    Estimate the value of each action from a given state.
    """
    _, nA = get_env_size(env)
    A = np.zeros(nA)

    for action in range(nA):
        for prob, next_state, reward, done in env.P[state][action]:
            A[action] += prob * (
                reward + discount_factor * V[next_state] * (not done)
            )

    return A


def policy_to_matrix(policy, nS, nA):
   
    policy_matrix = np.zeros((nS, nA))
    policy_matrix[np.arange(nS), policy] = 1.0
    return policy_matrix

'''Policy Iteration Program Part-2'''
def policy_iteration(env, theta=0.0001, discount_factor=1.0, max_iterations=1000):
    """
    Run Policy Iteration.

    Returns:
        policy_matrix: One-hot optimal policy
        V: Optimal value function
    """
    nS, nA = get_env_size(env)

    V = np.zeros(nS)
    

    # policy[state] = action chosen in that state
    policy = np.zeros(nS, dtype=int)

    for iteration in range(max_iterations):

        # -------------------------
        # Policy Evaluation
        # -------------------------
        while True:
            delta = 0

            for state in range(nS):
                old_value = V[state]
                action = policy[state]

                new_value = 0
                for prob, next_state, reward, done in env.P[state][action]:
                    new_value += prob * (
                        reward + discount_factor * V[next_state] * (not done)
                    )

                V[state] = new_value
                delta = max(delta, abs(old_value - new_value))

            if delta < theta:
                break

        # -------------------------
        # Policy Improvement
        # -------------------------
        policy_stable = True

        for state in range(nS):
            old_action = policy[state]

            action_values = action_estimation(env, state, V, discount_factor)
            best_action = np.argmax(action_values)

            policy[state] = best_action

            if old_action != best_action:
                policy_stable = False

        print("Iteration:", iteration + 1, "Delta:", delta)

        if policy_stable:
            print("Policy converged.")
            break

    policy_matrix = policy_to_matrix(policy, nS, nA)

    return policy_matrix, V


policy, V = policy_iteration(env)

print("Optimal Policy:")
print(policy)

print("Value Function:")
print(V)
		