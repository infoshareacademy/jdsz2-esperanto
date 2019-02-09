import numpy as np
import scipy.stats as stats

from projekt_python.generator import generator_liczb_losowych


def test_statystyczny(ile_liczb, ile_testow, xmin=0, xmax=1):
    count_eliminacja = 0
    count_random = 0

    for i in range(1, ile_testow):
        liczby_losowe_eliminacja = generator_liczb_losowych(xmin, xmax, ile_liczb)[0]
        liczby_losowe_random = np.random.uniform(size=ile_liczb)
        test_eliminacja = stats.kstest(liczby_losowe_eliminacja, 'uniform')
        test_random = stats.kstest(liczby_losowe_random, 'uniform')
        count_eliminacja += test_eliminacja.pvalue > 0.05
        count_random += test_random.pvalue > 0.05
    return count_eliminacja, count_random


if __name__ == '__main__':
    print(test_statystyczny(1000, 100))
