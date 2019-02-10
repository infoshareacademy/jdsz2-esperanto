import statistics
import sys
import time
import pickle
import numpy as np
from scipy.stats import norm, kstest, shapiro
from statsmodels.stats.diagnostic import lilliefors

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab



def generator_liczb_losowych(xmin, xmax, n):
    max_wartosc_funkcji = max([norm.pdf(c) for c in np.arange(xmin, xmax, (xmax-xmin)/n)])
    losowe = []

    while len(losowe) < n:
        r = np.random.uniform()
        x = np.random.uniform(low=xmin, high=xmax)
        if r < norm.pdf(x)/max_wartosc_funkcji:
            losowe.append(x)

    return losowe


# Test Lillieforsa
def L(x, alfa=0.05):
    D, p_l = lilliefors(x, 'norm')#, args=(0, 1))
    if p_l < alfa:
        return 0
    else:
        return 1


liczba_l = 0

u = 100
n = 3000

for i in range(n):
    i = generator_liczb_losowych(-3, 3, u)
    x = i
    L(x)
    liczba_l += L(x)

liczba_los = 0
for i in range(n):
    i = norm.rvs(size = u)
    x = i
    L(x)
    liczba_los += L(x)
sys.stdout = open('results_lillieforse.csv', 'a')
print('Test L dla {} losowan z generatora {} liczb: {}'.format(u, n, liczba_l/n))
print('Test L dla {} losowan z normalnego {} liczb: {}'.format(u, n, liczba_los/n))
sys.stdout.close()