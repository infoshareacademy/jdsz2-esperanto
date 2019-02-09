import statistics
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import norm, kstest, shapiro

"""
Tutaj możemy sobie zilustrowac nasze wygenerowane liczby na histogramie wraz z dopasowanym 
rozkladem normalnym
"""

def generator_liczb_losowych(xmin, xmax, n):
    max_wartosc_funkcji = max([norm.pdf(c) for c in np.arange(xmin, xmax, (xmax-xmin)/n)])
    losowe = []
    count = 0
    while len(losowe) < n:
        r = np.random.uniform()
        x = np.random.uniform(low=xmin, high=xmax)
        if r < norm.pdf(x)/max_wartosc_funkcji:
            losowe.append(x)
        else:
            count += 1
    return losowe, count

x, b =generator_liczb_losowych(-3,3,1000)
print('Liczba losowan, ktorej wynikiem bylo {} liczb: {}'.format(len(x),b))

(mu, sigma) = norm.fit(x)
n, bins, patches = plt.hist(x, 60, density=1)
y = norm.pdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth = 2)
plt.ylabel('y')
plt.xlabel('X')
plt.title('Histogram wygenerowanych liczb z dopasowanym rozkładem normalnym')
plt.show()