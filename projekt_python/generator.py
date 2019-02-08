import statistics
import sys
import time

import numpy as np
from scipy.stats import norm, kstest


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
