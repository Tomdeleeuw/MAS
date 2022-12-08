import numpy as np
from scipy import stats
import math
import random
import matplotlib.pyplot as plt

def mc_sim1():
    sample_size=100000
    X = np.random.normal(0, 1, sample_size)
    print(X)
    formula = np.cos(X) ** 2
    # formula = X
    mean = np.mean(formula)
    se = np.std(formula) / np.sqrt(sample_size)
    print(mean, se)

def mc_sim2():
    sample_size=10000
    n = 10
    rho_obs = 0.3
    rho_sim = np.zeros(sample_size)
    for i in range(sample_size):
        S = np.random.normal(0,1,n)
        A = np.random.normal(0,1,n)
        rho = np.corrcoef(S,A)
        rho_sim[i] = rho[0,1]
    rho_sim=np.sort(rho_sim)
    p_value = len(np.where(rho_sim>rho_obs)[0])/len(rho_sim)
    print(p_value)

def mc_sim3():
    sample_size=100000
    X = np.random.uniform(-5,5,sample_size)
    f = stats.norm.pdf(X)
    g = 1/10
    formula = X**2
    F = formula*f/g
    mean = np.mean(F)
    se = np.std(F) / np.sqrt(sample_size)
    print(mean, se)

def mc_sim4():
    sample_size=100000
    X = np.random.uniform(-1,1,sample_size)
    f = (1+np.cos(math.pi*X))/2
    g = 1/2
    formula = X**2
    F = formula*f/g
    mean = np.mean(F)
    se = np.std(F) / np.sqrt(sample_size)
    print(mean, se)

def bandit(mu):
    return random.gauss(mu, 1)

def kbandit(k):
    mus = [random.uniform(-3, 4) for _ in range(k)]
    q_a = np.zeros(len(k))
    n_a = np.zeros(len(k))

    




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mc_sim4()
