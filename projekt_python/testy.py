import statistics
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import norm, kstest, shapiro


# Testy Shapiro-Wilka i Kolmogorova-Smirnova dla losowych liczb z rozkładu normalnego
# hipoteza H0: rozklad zmiennej jest rozkladem normalnym
# hipoteza H1: rozklad zmiennej nie jest rozkladem normalnym

# Jesli p-value > poziomu istotnosci przyjetego przez nas,
#           wowczas przyjmujemy hipoteze H0,
#           zatem rozklad naszej zmiennej jest rozkladem normalnym
# Jesli p-value < poziomu istotnosci przyjetego przez nas,
#           wowczas odrzucamy H0 na korzysc H1,
#           zatem rozklad naszej zmiennej nie jest rozkladem normalnym



x= norm.rvs(size = 1000)
#alfa to przyjety przez nas poziom istotnosci, domyslnie wynosi 0,05
def SW(x,alfa=0.05):
    W, p_sw = shapiro(x)
    #print('Test Shapiro-Wilka:\n p-value = ',p_sw)
    if p_sw < alfa:
        return 0
        #print('Badany rozklad nie jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)
    return 1
        #print('Badany rozklad jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)


def KS(x,alfa=0.05):
    D, p_ks = kstest(x, 'norm', args=(np.mean(x), np.std(x, ddof=1)))
    #print('Test Kolmogorova-Smirnova:\n p-value = ',p_ks)
    if p_ks < alfa:
        return 0
        #print('Badany rozklad nie jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)
    return 1
        #print('Badany rozklad jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)

#print('Kolmogorova-Smirnova:\n p-value = {}\n Odrzucic hipoteze zerowa? {}'.format(p_ks, p_ks < 0.05))
#print('SW: {}, KS: {}'.format(p_sw, p_ks))
# liczba = 0
# n = 10000
# for i in range(n):
#     i = norm.rvs(size = 1000)
#
#     x = i
#     KS(x)
#     liczba += KS(x)
# print(liczba)
# print(float(liczba/n))

#print('Test S-W:', SW(x, alfa=0.1))
#print()
#print('Test K-S:', KS(x))


(mu, sigma) = norm.fit(x)
n, bins, patches = plt.hist(x, 60, density=1)
y = norm.pdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth = 2)
plt.ylabel('y')
plt.xlabel('X')
plt.title('Histogram wygenerowanych liczb z dopasowanym rozkładem normalnym')
plt.show()
