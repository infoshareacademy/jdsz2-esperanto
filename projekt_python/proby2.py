import numpy as np
from scipy.stats import norm, kstest, shapiro
from projekt_python.generator import generator_liczb_losowych


def SW(x, alfa=0.05):
    W, p_sw = shapiro(x)
    # print('Test Shapiro-Wilka:\n p-value = ',p_sw)
    if p_sw < alfa:
        return 0
        # print('Badany rozklad nie jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)
    return 1
        # print('Badany rozklad jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)


def KS(x, alfa=0.05):
    D, p_ks = kstest(x, 'norm', args=(np.mean(x), np.std(x, ddof=1)))
    # print('Test Kolmogorova-Smirnova:\n p-value = ',p_ks)
    if p_ks < alfa:
        return 0
        # print('Badany rozklad nie jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)
    return 1
        # print('Badany rozklad jest rozkladem normalnym na poziomie istotnosci %a.'%alfa)

# print('Kolmogorova-Smirnova:\n p-value = {}\n Odrzucic hipoteze zerowa? {}'.format(p_ks, p_ks < 0.05))
# print('SW: {}, KS: {}'.format(p_sw, p_ks))


if __name__ == '__main__':
    liczba_ks = 0
    liczba_sw = 0
    n = 10
    for i in range(n):
        i = generator_liczb_losowych(-3, 3, 1000)  # FIXME: dlaczego nadpisujesz tutaj zmienną i?
        x = i
        KS(x)
        SW(x)
        liczba_ks += KS(x)
        liczba_sw += SW(x)

    print('Test KS dla {} prob: {}'.format(n, liczba_ks/n))
    print('Test SW dla {} prob: {}'.format(n, liczba_sw/n))