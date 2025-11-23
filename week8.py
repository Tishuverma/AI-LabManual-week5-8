#week 8
# gridworld_mdp.py
import numpy as np

class Gridworld:
    """
    Gridworld MDP.
    - shape: (rows, cols)
    - terminals: dict {(r,c): reward}
    - rewards: default reward per step (can be overridden per cell via rewards_map)
    - p_success: probability of executing intended action; remaining prob split to sideways actions
    - gamma: discount factor
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(self, shape=(4,3), terminals=None, rewards_map=None,
                 default_reward=-0.04, p_success=0.8, gamma=0.99):
        self.rows, self.cols = shape
        self.nS = self.rows * self.cols
        self.nA = len(self.ACTIONS)
        self.terminals = terminals or {}
        self.rewards_map = rewards_map or {}
        self.default_reward = default_reward
        self.p_success = p_success
        self.gamma = gamma

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def state_idx(self, r, c):
        return r * self.cols + c

    def idx_state(self, idx):
        return divmod(idx, self.cols)

    def reward(self, r, c):
        return self.rewards_map.get((r,c), self.default_reward)

    def transition_distribution(self, s, a):
        """
        Returns list of (prob, next_state, reward, done) for state index s and action a.
        uses slip to sideways moves.
        """
        r, c = self.idx_state(s)
        if (r, c) in self.terminals:
            # terminal: stays in terminal with reward (absorbing)
            return [(1.0, s, self.terminals[(r,c)], True)]

        # intended movement
        dr, dc = self.ACTIONS[a]
        # sideways actions (left and right relative)
        left = (a - 1) % 4
        right = (a + 1) % 4

        outcomes = []
        for prob, act in [(self.p_success, a), ((1-self.p_success)/2, left), ((1-self.p_success)/2, right)]:
            dr2, dc2 = self.ACTIONS[act]
            nr, nc = r + dr2, c + dc2
            if not self.in_bounds(nr, nc):
                # bounce (stay)
                ns = s
            else:
                ns = self.state_idx(nr, nc)
            done = (self.idx_state(ns) in self.terminals)
            rew = self.terminals.get(self.idx_state(ns), self.reward(*self.idx_state(ns)))
            outcomes.append((prob, ns, rew, done))
        # combine duplicate next states
        combined = {}
        for prob, ns, rew, done in outcomes:
            key = (ns, rew, done)
            combined[key] = combined.get(key, 0.0) + prob
        return [(p, ns, rew, done) for (ns, rew, done), p in combined.items()]

    def value_iteration(self, theta=1e-6, max_iters=10000):
        V = np.zeros(self.nS)
        for idx,(r,c) in enumerate([self.idx_state(i) for i in range(self.nS)]):
            if (r,c) in self.terminals:
                V[idx] = self.terminals[(r,c)]
        it = 0
        while it < max_iters:
            delta = 0.0
            for s in range(self.nS):
                r, c = self.idx_state(s)
                if (r,c) in self.terminals:
                    continue
                action_values = np.zeros(self.nA)
                for a in range(self.nA):
                    for prob, ns, rew, done in self.transition_distribution(s,a):
                        action_values[a] += prob * (rew + self.gamma * V[ns])
                maxv = np.max(action_values)
                delta = max(delta, abs(maxv - V[s]))
                V[s] = maxv
            it += 1
            if delta < theta:
                break
        # derive greedy policy
        policy = np.zeros(self.nS, dtype=int)
        for s in range(self.nS):
            r, c = self.idx_state(s)
            if (r,c) in self.terminals:
                policy[s] = -1
                continue
            action_values = np.zeros(self.nA)
            for a in range(self.nA):
                for prob, ns, rew, done in self.transition_distribution(s,a):
                    action_values[a] += prob * (rew + self.gamma * V[ns])
            policy[s] = np.argmax(action_values)
        return V.reshape((self.rows, self.cols)), policy.reshape((self.rows, self.cols)), it

    def policy_evaluation(self, policy, theta=1e-6):
        V = np.zeros(self.nS)
        for idx,(r,c) in enumerate([self.idx_state(i) for i in range(self.nS)]):
            if (r,c) in self.terminals:
                V[idx] = self.terminals[(r,c)]
        while True:
            delta = 0.0
            for s in range(self.nS):
                r, c = self.idx_state(s)
                if (r,c) in self.terminals:
                    continue
                a = policy[s]
                v = 0.0
                for prob, ns, rew, done in self.transition_distribution(s,a):
                    v += prob * (rew + self.gamma * V[ns])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < theta:
                break
        return V

    def policy_iteration(self, max_iters=1000):
        # initialize random policy
        policy = np.zeros(self.nS, dtype=int)
        for s in range(self.nS):
            if self.idx_state(s) in self.terminals:
                policy[s] = -1
            else:
                policy[s] = np.random.randint(0, self.nA)
        it = 0
        while it < max_iters:
            it += 1
            V = self.policy_evaluation(policy)
            policy_stable = True
            for s in range(self.nS):
                r, c = self.idx_state(s)
                if (r,c) in self.terminals:
                    continue
                old_action = policy[s]
                action_values = np.zeros(self.nA)
                for a in range(self.nA):
                    for prob, ns, rew, done in self.transition_distribution(s,a):
                        action_values[a] += prob * (rew + self.gamma * V[ns])
                best_a = np.argmax(action_values)
                policy[s] = best_a
                if old_action != best_a:
                    policy_stable = False
            if policy_stable:
                break
        return V.reshape((self.rows, self.cols)), policy.reshape((self.rows, self.cols)), it


if __name__ == "__main__":
    # Example: 4x3 grid like Sutton & Barto with two terminal states
    terminals = {(0, 3): +1.0, (1, 3): -1.0}  # put terminals by (row,col) indexing from top-left (0,0)
    rewards_map = {}  # empty: use default per-step reward
    gw = Gridworld(shape=(3,4), terminals=terminals, default_reward=-0.04, p_success=0.8, gamma=0.99)

    V_vi, policy_vi, it_vi = gw.value_iteration()
    print("Value Iteration finished in", it_vi, "iterations.")
    print("V (rows x cols):\n", np.round(V_vi, 3))
    print("Policy (0:Up,1:Right,2:Down,3:Left; -1=terminal):\n", policy_vi)

    V_pi, policy_pi, it_pi = gw.policy_iteration()
    print("\nPolicy Iteration finished in", it_pi, "iterations.")
    print("V (rows x cols):\n", np.round(V_pi, 3))
    print("Policy (0:Up,1:Right,2:Down,3:Left; -1=terminal):\n", policy_pi)
