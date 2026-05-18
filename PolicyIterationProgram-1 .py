!pip install --user --upgrade gym==0.20.0

import numpy as np
import pprint
from gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)

env = GridworldEnv()


def action_estimation(env, state, V, discount_factor=1.0):
    A = np.zeros(env.nA)

    for action in range(env.nA):
        for prob, next_state, reward, done in env.P[state][action]:
            A[action] += prob * (
                reward + discount_factor * V[next_state] * (not done)
            )

    return A


def policy_estimation(env, V, discount_factor=1.0):
    policy = np.ones((env.nS, env.nA)) / env.nA

    while True:
        policy_stable = True

        for state in range(env.nS):
            old_action = np.argmax(policy[state])

            action_values = action_estimation(env, state, V, discount_factor)
            best_action = np.argmax(action_values)

            if best_action != old_action:
                policy_stable = False

            policy[state, :] = 0
            policy[state, best_action] = 1.0

        if policy_stable:
            return policy

		