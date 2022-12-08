import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

n = 10000
n_sim = 1000
pcts = 90


def search(n, pcts):
    X = list(np.linspace(0, 1, n))
    N = list(range(n))
    random.shuffle(X)
    random.shuffle(N)
    best_house = N[X.index(max(X))]
    chosen_houses = np.zeros(pcts)
    for i in range(pcts):
        sample_size = int(np.floor((i+1) / 100 * n))
        best = max(X[:sample_size])
        visit = N[sample_size:]
        visit_X = X[sample_size:]
        for j in visit:
            if visit_X[visit.index(j)] > best:
                chosen_houses[i] = j
                break
    right_choice = chosen_houses == best_house
    return right_choice


t0 = time.time()
results = pd.DataFrame(data=None, columns=list(range(1, pcts+1)))
for i in range(n_sim):
    it = search(n, pcts)
    results = results.append(pd.Series(data=it, index=list(range(1, pcts+1))), ignore_index=True)
results.to_csv('results_test.csv')
print(results)
t1 = time.time()
print(t1-t0)



