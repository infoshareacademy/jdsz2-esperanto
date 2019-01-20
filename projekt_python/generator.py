import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt


def generator_liczb_losowych(xmin, xmax, n):
    max_wartosc_funkcji = max([norm.pdf(c) for c in np.arange(xmin, xmax, (xmax-xmin)/n)])
    print(max_wartosc_funkcji, [x for x in np.arange(xmin, xmax, (xmax-xmin)/n)])
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
    # TODO: zmienia się norm.pdf i x = funkcja

if __name__ == '__main__':
    wynik, count = generator_liczb_losowych(-3, 3, 1000)
    print(wynik)
    print(count)
    plt.hist(wynik)
    plt.show()
