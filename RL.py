import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import random
import time

legend = {0: 'empty', 1: 'wall', 2: 'pit', 3: 'treasure'}


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.q = [0, 0, 0, 0]
        self.r = None
        self.v = None
        self.c = None
        self.e = None
        self.w = None
        self.n = None
        self.s = None


class Grid:
    def __init__(self, w, h, policy, treasure, pit, obstacles):
        self.w = w
        self.h = h
        self.treasure_loc = treasure
        self.treasure = []
        self.pit_loc = pit
        self.pit =[]
        self.plot_grid = None
        self.index = range(1, h+1)
        self.columns = range(1, w+1)
        self.policy = policy
        self.obstacles_loc = obstacles
        self.obstacles = []
        self.grid = self.populate()
        self.agent_loc = (9, 1)

    def populate(self):
        grid = pd.DataFrame(index=self.index, columns=self.columns)
        for i in self.index:
            for j in self.columns:
                grid[i][j] = State(i, j)
        for state in self.obstacles_loc:
            grid[state[0]][state[1]].c = 'wall'
            self.obstacles.append(grid[state[0]][state[1]])
        for state in self.pit_loc:
            grid[state[0]][state[1]].c = 'pit'
            self.pit.append(grid[state[0]][state[1]])
        for state in self.treasure_loc:
            grid[state[0]][state[1]].c = 'treasure'
            self.treasure.append(grid[state[0]][state[1]])
        for i in self.index:
            for j in self.columns:
                grid[i][j].r = -1
                if j != 1:
                    grid[i][j].w = grid[i][j-1]
                if j != self.w:
                    grid[i][j].e = grid[i][j+1]
                if i != 1:
                    grid[i][j].n = grid[i-1][j]
                if i != self.h:
                    grid[i][j].s = grid[i+1][j]
                if grid[i][j].c == 'treasure':
                    grid[i][j].v = 50
                    grid[i][j].r = 50
                    self.treasure = grid[i][j]
                elif grid[i][j].c == 'pit':
                    grid[i][j].v = -50
                    grid[i][j].r = -50
                    self.pit = grid[i][j]
                elif grid[i][j].c == 'wall':
                    pass
                    # grid[i][j].v = -1
                else:
                    grid[i][j].v = -1
                    grid[i][j].c = ''
        return grid

    def move(self, state, action=None):
        print(action)
        if action is None:
            next_state = self.choose(state)
        else:
            if action == 0:
                next_state = state.n
            elif action == 1:
                next_state = state.w
            elif action == 2:
                next_state = state.s
            elif action == 3:
                next_state = state.e
            else:
                raise Exception(str(action), "is not a valid action")
        if next_state is None:
            next_state = state
        elif next_state in self.obstacles:
            next_state = state
        reward = next_state.v
        return next_state, reward

    def choose(self, state):
        if self.policy == 'eq':
            next_state = random.choice([state.w, state.n, state.s, state.e])
            return next_state
            # return random.choice([x for x in [state.w, state.n, state.s, state.e] if x is not None])

    def evaluate(self, MC):
        for it in range(MC):
            i = random.choice(list(range(1, self.h+1)))
            j = random.choice(list(range(1, self.w+1)))
            state = self.grid[i][j]
            if state.c != '':
                pass
            else:
                rewards = [x.r for x in [state.w, state.n, state.s, state.e] if x is not None]
                if len(rewards) < 4:
                    [rewards.append(-1) for x in [state.w, state.n, state.s, state.e] if x is None]
                values = [x.v for x in [state.w, state.n, state.s, state.e] if x is not None and x.c != 'wall']
                if len(values) < 4:
                    [values.append(state.v) for x in [state.w, state.n, state.s, state.e] if x is None]
                state.v = np.average(rewards) + np.average(values)

    def plot(self, title, MC):
        self.plot_grid = pd.DataFrame(index=self.index, columns=self.columns)
        for i in self.index:
            for j in self.columns:
                self.plot_grid[i][j] = self.grid[i][j].v
        for column in self.plot_grid:
            self.plot_grid[column] = pd.to_numeric(self.plot_grid[column])
        cmap = plt.cm.summer
        fig = plt.figure(figsize=(10,6))
        ax = sns.heatmap(self.plot_grid.values, cmap=cmap)
        ax.tick_params(axis='y', rotation=0)
        plt.title((f"{title}" + "\n") + f"state-value function after {MC} episodes")
        plt.savefig(f"{title}.png")
        plt.show()

    def plot_q(self, title='', episode_num=100):
        fig = plt.figure(figsize=(10,6))

        cmap = plt.cm.summer

        self.plot_grid = pd.DataFrame(index=self.index, columns=self.columns)
        for i in self.index:
            for j in self.columns:
                self.plot_grid[i][j] = max(self.grid[i][j].q)
        for column in self.plot_grid:
            self.plot_grid[column] = pd.to_numeric(self.plot_grid[column])
        self.plot_grid = self.plot_grid.transpose()
        v_table_optimal = np.array(self.plot_grid.values.tolist())
        v_table_optimal.reshape(self.h, self.w)
        ax = plt.imshow(v_table_optimal, cmap=cmap)

        for x in list(range(self.w)):
            for y in list(range(self.h)):
                q_values = np.array(self.grid[x + 1][y + 1].q)
                q_values = q_values - np.min(q_values)
                q_norm = q_values / np.sum(q_values)

                optimal_q = q_norm.max()
                for i, q in enumerate(q_norm):
                    if q < optimal_q:  # if q < 0.3:
                        continue
                    if i == 0:  # UP is inverted because of imshow
                        plt.arrow(y, x, 0, -0.5 * q, fill=False,
                                  length_includes_head=True, head_width=0.1,
                                  alpha=0.8, color='k')
                    if i == 1:  # LEFT
                        plt.arrow(y, x, -0.5 * q, 0, fill=False,
                                  length_includes_head=True, head_width=0.1,
                                  alpha=0.8, color='k')
                    if i == 2:  # DOWN
                        plt.arrow(y, x, 0, 0.5 * q, fill=False,
                                  length_includes_head=True, head_width=0.1,
                                  alpha=0.8, color='k')
                    if i == 3:  # RIGHT
                        plt.arrow(y, x, 0.5 * q, 0, fill=False,
                                  length_includes_head=True, head_width=0.1,
                                  alpha=0.8, color='k')

        plt.title((f"{title}" + "\n") + f"Optimal policy found after {episodes} episodes")
        plt.colorbar(ax, orientation='vertical')
        plt.savefig(f'{title}.png')
        plt.show()

    def __str__(self):
        fmt = "g"
        self.plot_grid = pd.DataFrame(index=self.index, columns=self.columns)
        for i in self.index:
            for j in self.columns:
                self.plot_grid[i][j] = self.grid[i][j].v

        col_maxes = [max([len(f"{x}") for x in self.plot_grid[col]]) for col in
                     self.plot_grid.columns]
        for x, z in self.plot_grid.T.iteritems():
            for i, y in enumerate(z):
                if not pd.isna(y):
                    print(("{:" + str(col_maxes[i]-10) + fmt + "}").format(y), end="  ")
                else:
                    print(("{:" + str(col_maxes[i]-10) + fmt + "}").format(0), end="  ")
            print("")
        return f"{self.h} x {self.w} Gridworld"


class SARSA:
    def __init__(self, env, eps, alpha, gamma):
        self.env = env
        self.q_table = np.zeros((env.h, env.w, 4))
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.total_reward = 0
        self.terminal = [self.env.treasure, self.env.pit]

    def start_state(self):
        while True:
            h = random.randint(1, self.env.h)
            w = random.randint(1, self.env.w)
            state = self.env.grid[h][w]
            if state not in self.env.obstacles:
                return state

    def action(self, state):
        p = random.uniform(0, 1)
        if p > self.eps:
            max_q = np.max(state.q)
            max_actions = [i for i, q in enumerate(state.q) if q == max_q]
            action = random.choice(max_actions)
        else:
            action = random.randint(0, 3)
        return action

    def update(self, state, action, next_state, next_action):
        new = next_state.r + self.gamma * next_state.q[next_action]
        old = new - state.q[action]
        state.q[action] += self.alpha * old

    def run(self, episodes):
        for i in range(episodes):
            print('New episode:', i)
            self.total_reward = 0
            state = self.start_state()
            print('Start state:', '('+str(state.x) + ',' + str(state.y)+')')
            self.env.agent_loc = state
            action = self.action(state)
            while state not in self.terminal:
                next_state, reward = self.env.move(state, action)
                print('next state:', '(' + str(next_state.x) + ',' + str(next_state.y) + ')')
                next_action = self.action(next_state)
                self.update(state, action, next_state, next_action)
                state, action = next_state, next_action
                self.total_reward += reward
            print(self.total_reward)


class QL(SARSA):
    def update_ql(self, state, action, next_state, next_action):
        new = next_state.r + self.gamma * np.max(next_state.q)
        old = new - state.q[action]
        state.q[action] += self.alpha * old

    def run_ql(self, episodes):
        for i in range(episodes):
            print('New episode:', i)
            self.total_reward = 0
            state = self.start_state()
            print('Start state:', '('+str(state.x) + ',' + str(state.y)+')')
            self.env.agent_loc = state
            action = self.action(state)
            while state not in self.terminal:
                next_state, reward = self.env.move(state, action)
                print('next state:', '(' + str(next_state.x) + ',' + str(next_state.y) + ')')
                next_action = self.action(next_state)
                self.update_ql(state, action, next_state, next_action)
                state, action = next_state, next_action
                self.total_reward += reward
            print(self.total_reward)

if __name__ == '__main__':
    obstacles = [(3,2), (4,2), (5,2), (6,2), (7,2), (7,3), (7,4), (7,5), (7,6), (2,8), (3,8), (4,8), (5,8)]
    treasure = [(9,9)]
    pit = [(6,7)]
    eps = 0.1
    alpha = 0.1
    gamma = 0.9
    episodes = 1000
    MC = 100000
    h = 9
    w = 9

    t0 = time.time()
    gridworld = Grid(h, w, 'eq', treasure, pit, obstacles)
    print(gridworld)
    gridworld.evaluate(MC)
    gridworld.plot('MC policy evaluation', MC)

    gridworld = Grid(h, w, 'eq', treasure, pit, obstacles)
    Sarsa = SARSA(gridworld, eps, alpha, gamma)
    print(Sarsa.run(episodes))
    gridworld.plot_q(title='SARSA', episode_num=episodes)
    t1 = time.time()

    gridworld = Grid(h, w, 'eq', treasure, pit, obstacles)
    QL = QL(gridworld, eps, alpha, gamma)
    print(QL.run(episodes))
    print('Total runtime:', t1 - t0)
    gridworld.plot_q(title='Q-Learning', episode_num=episodes)
