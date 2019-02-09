import statistics
import sys
import time
import numpy as np
from scipy.stats import norm, kstest, shapiro
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab



def generator_liczb_losowych(xmin, xmax, n):
    max_wartosc_funkcji = max([norm.pdf(c) for c in np.arange(xmin, xmax, (xmax-xmin)/n)])
    # print(max_wartosc_funkcji, [x for x in np.arange(xmin, xmax, (xmax-xmin)/n)])
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



def SW(x,alfa=0.05):
    W, p_sw = shapiro(x)
    #print('Test Shapiro-Wilka:\n p-value = ',p_sw)
    if p_sw < alfa:
        return 0
        #print('Badany rozklad nie jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)
    else:
        return 1
        #print('Badany rozklad jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)


def KS(x,alfa=0.05):
    D, p_ks = kstest(x, 'norm', args=(np.mean(x), np.std(x, ddof=1)))
    #print('Test Kolmogorova-Smirnova:\n p-value = ',p_ks)
    if p_ks < alfa:
        return 0
        #print('Badany rozklad nie jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)
    else:
        return 1
        #print('Badany rozklad jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)

#print('Kolmogorova-Smirnova:\n p-value = {}\n Odrzucic hipoteze zerowa? {}'.format(p_ks, p_ks < 0.05))
#print('SW: {}, KS: {}'.format(p_sw, p_ks))

liczba = 0
n = 10
for i in range(0,n,1):
    i, miss_count = generator_liczb_losowych(-3, 3, 1000)
    x = i
    KS(x)
    liczba += KS(x)
    valid = liczba/n

print(liczba, valid)