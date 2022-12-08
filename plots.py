import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

n = 10000
n_sim = 1000
pcts = 90

results = pd.read_csv('results.csv')
print(results)
probabilities = list(sum(results[str(i)]) / n_sim for i in range(1, pcts + 1))

maxprob = max(probabilities)
maxindex = probabilities.index(maxprob)
print(maxindex)

textstr = '\n'.join((
    'probability={0}'.format(maxprob),
    r'sample size={0}%'.format(maxindex)))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

fig, ax1 = plt.subplots()
ax1.plot(probabilities, color='black')
ax1.text(0.72, 0.97, textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.title('Probability of finding the best house')
plt.xlabel("Percentage of houses in sample")
plt.ylabel('Probability')
plt.xticks(np.linspace(0, pcts, 10))
plt.axhline(maxprob, 0, 0.03)
plt.axvline(maxindex, 0, 0.05)
plt.savefig('Probs')
plt.show()
