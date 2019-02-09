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
    # print(max_wartosc_funkcji, [x for x in np.arange(xmin, xmax, (xmax-xmin)/n)])
    losowe = []

    while len(losowe) < n:
        r = np.random.uniform()
        x = np.random.uniform(low=xmin, high=xmax)
        if r < norm.pdf(x)/max_wartosc_funkcji:
            losowe.append(x)

    return losowe



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

liczba_ks = 0
liczba_sw = 0
n = 40
u = 1000

for i in range(n):
    #i = generator_liczb_losowych(-3, 3, u)  # FIXME: dlaczego nadpisujesz tutaj zmienną i?
    i = norm.rvs(size=u)
    x = i
    KS(x)
    SW(x)
    liczba_ks += KS(x)
    liczba_sw += SW(x)

sys.stdout = open('results.csv', 'a')
#print('generator Ko-Sm, {}, {}, {}'.format(n, u, liczba_ks/n))
#print('generator Sh-Wi, {}, {}, {}'.format(n, u, liczba_sw/n))
print('losowe Ko-Sm, {}, {}, {}'.format(n, u, liczba_ks/n))
print('losowe Sh-Wi, {}, {}, {}'.format(n, u, liczba_sw/n))
sys.stdout.close()