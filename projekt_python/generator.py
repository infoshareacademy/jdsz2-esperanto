import statistics
import sys
import time

import numpy as np
from scipy.stats import norm, kstest, shapiro


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

print(generator_liczb_losowych(-3,-3,1000))

"""
if __name__ == '__main__':
    count = 50
    ds = []
    ps = []
    while count > 0:
        wynik, miss_count = generator_liczb_losowych(-3, 3, 1000)
        d, p = kstest(wynik, 'norm')
        ds.append(d)
        ps.append(p)
        count = count - 1
        time.sleep(1)
        sys.stdout.write("\rZostało jeszcze kroków: %d" % count)
        sys.stdout.flush()
    print("\n", statistics.mean(ds), statistics.mean(ps))
    print(wynik)
    print(len(wynik))
"""

# Testy Shapiro-Wilka i Kolmogorova-Smirnova dla losowych liczb z rozkładu normalnego

#x= norm.rvs(size = 1000)
#W, p_sw = shapiro(x)
#D, p_ks = kstest(x, 'norm', args=(np.mean(x), np.std(x, ddof=1)))

#print('SW: {}, KS: {}'.format(p_sw, p_ks))