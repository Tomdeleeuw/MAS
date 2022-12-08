import numpy as np
import pandas as pd
import random
import time
import math


class Node:
    def __init__(self, val):
        self.l = None
        self.r = None
        self.parent = None
        self.address = None
        self.visits = 0
        self.level = 0
        self.v = val
        self.UCB = 0


class Tree:
    def __init__(self):
        self.root = None
        self.levels = None
        self.addresses = None

    def getRoot(self):
        return self.root

    def populate(self, d):
        self.root = Node(0)
        self.root.address = ''
        self.levels = {0: [self.root]}
        self.addresses = {0: [self.root.address]}
        for i in range(d):
            level = []
            addresses = []
            for j in self.levels[i]:
                j.l = Node(0)
                j.l.address = j.address + 'L'
                j.l.parent = j
                j.l.level = i+1
                j.r = Node(0)
                j.r.address = j.address + 'R'
                j.r.parent = j
                j.r.level = i+1
                level.append(j.l)
                level.append(j.r)
                addresses.append(j.l.address)
                addresses.append(j.r.address)
            self.levels[i+1] = level
            self.addresses[i+1] = addresses

    def depth(self):
        return len(self.levels)-1

    def distance(self, node, target):
        if self.root is None:
            return None
        address = [x for x in node.address]
        address_target = [x for x in target.address]
        dist = len(address) - sum([address[i] == address_target[i] for i in range(len(address))])
        return dist

    def deleteTree(self):
        self.root = None

    def __str__(self):
        pass


class MCTS:
    def __init__(self, d, B, t, c):
        self.budget = 50
        self.rollouts = 5
        self.c = c
        self.tree = Tree()
        self.tree.populate(d)
        self.target_node = random.choice(self.tree.levels[d])
        self.snowcap = []
        self.solution = None

        for node in self.tree.levels[d]:
            distance = self.tree.distance(node, self.target_node)
            node.v = B*math.exp(-distance/t)

    def select(self, node):
        if node.level == self.tree.depth():
            print("Leaf node reached")
            return node
        elif node.level == self.tree.depth()-1:
            if node.l.v > node.r.v:
                return node.l
            else:
                return node.r
        else:
            # return random.choice(self.snowcap)
            max_ucb = np.argmax([x.UCB for x in self.snowcap])
            return self.snowcap[max_ucb]

    def expansion(self, node):
        node.visits += 1
        for child in [node.l, node.r]:
            if child not in self.snowcap:
                self.snowcap.append(child)
            child.visits += 1
            rollouts = 1
            sims = []
            while rollouts <= self.rollouts:
                leaf_node = self.simulate(child)
                sims.append(leaf_node.v)
                rollouts += 1
            child.v = np.average(sims)
            self.backup(child, np.average(sims))
            self.policy(child)

    def simulate(self, node):
        next_node = random.choice([node.l, node.r])
        while next_node.level < self.tree.depth():
            next_node = self.simulate(next_node)
        return next_node

    def backup(self, node, value):
        parent_node = node.parent
        parent_node.v += value
        parent_node.visits += 1
        while parent_node.level > 0:
            self.policy(parent_node)
            parent_node = self.backup(parent_node, value)
        return parent_node

    def search(self, node):
        it = 1
        while it <= self.budget:
            self.expansion(node)
            it += 1
        new_root = self.select(node)
        self.snowcap.remove(new_root)
        if new_root.level < self.tree.depth()-1:
            self.search(new_root)
        else:
            final_node = self.select(new_root)
            print("MCTS Done. Results:")
            print('Target node address: ', self.target_node.address)
            print('Optimal Node address:', final_node.address)
            print(f'Score: {final_node.v:9.1f}/10.0')
            self.solution = final_node
        return self.solution

    def policy(self, node):
        node.UCB = node.v/node.visits + self.c*math.sqrt(np.log(node.parent.visits)/node.visits)


if __name__ == '__main__':
    d = 15
    B = 10
    tao = 5.25
    c = 2

    t0 = time.time()
    mcts = MCTS(d, B, tao, c)
    solution = mcts.search(mcts.tree.root)
    print(solution.v)
    t1 = time.time()
    print(f'Runtime:{t1-t0:9.2f} seconds')

    def experiment1():
        data = {}
        for i in [1, 5, 10]:
            solutions = []
            for j in range(100):
                mcts = MCTS(d, B, tao, c)
                mcts.rollouts = i
                solutions.append(mcts.search(mcts.tree.root).v)
            print('c:', i, 'average:', np.average(solutions))
            data[i] = np.average(solutions)
        return data

    def experiment2():
        data = {}
        for i in [0, 0.5, 1, 2, 5]:
            solutions = []
            runtime = []
            for j in range(100):
                mcts = MCTS(d, B, tao, c)
                mcts.c = i
                t0 = time.time()
                solutions.append(mcts.search(mcts.tree.root).v)
                t1 = time.time()
                runtime.append(t1-t0)
            print('c:', i, 'average:', np.average(solutions), 'runtime:', np.average(runtime))
            data[i] = np.average(solutions)
        return data


    # print(experiment2())





