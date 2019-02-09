import statistics
import sys
import time
import pickle
import numpy as np
from scipy.stats import norm, kstest, shapiro
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


#Test Shapiro-Wilka
def SW(x,alfa=0.05):
    W, p_sw = shapiro(x)
    if p_sw < alfa:
        return 0
    else:
        return 1

# Test Kolmogorova-Smirnova
def KS(x,alfa=0.05):
    D, p_ks = kstest(x, 'norm', args=(np.mean(x), np.std(x, ddof=1)))
    if p_ks < alfa:
        return 0
    else:
        return 1


liczba_ks = 0
liczba_sw = 0
n = 10
u = 1000

for i in range(n):
    i = generator_liczb_losowych(-3, 3, u)
    x = i
    KS(x)
    SW(x)
    liczba_ks += KS(x)
    liczba_sw += SW(x)

sys.stdout = open('results.csv', 'a')
print('Test KS dla {} losowan {} liczb: {}'.format(n, u, liczba_ks/n))
print('Test SW dla {} losowan {} liczb: {}'.format(n, u, liczba_sw/n))
sys.stdout.close()