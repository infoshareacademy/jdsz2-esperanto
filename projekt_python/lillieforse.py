import numpy as np
from scipy.stats import norm
from statsmodels.stats.diagnostic import lilliefors
# TODO: Po skończeniu pisania programu warto wcisnąć CTRL ALT O, celem zoptymalizowania importów - usuwa nieużywane
#  impoty i ładnie segreguje pozostałe. Ponadgo jak istalujesz jakąś zewnętrzną bibliotekę to od razu dodaj ją do
#  requirements.txt, dzięki czemu jak będziesz chciała w przyszłości odpalić projekt to wystarczy wpisać pip install
#  -r requirements.txt.


def generator_liczb_losowych(xmin, xmax, n):
    max_wartosc_funkcji = max([norm.pdf(c) for c in np.arange(xmin, xmax, (xmax-xmin)/n)])
    losowe = []

    while len(losowe) < n:
        r = np.random.uniform()
        x = np.random.uniform(low=xmin, high=xmax)
        if r < norm.pdf(x)/max_wartosc_funkcji:
            losowe.append(x)

    return losowe


def test_lillieforsa(x, alfa=0.05):
    D, p_l = lilliefors(x, 'norm')#, args=(0, 1))
    if p_l < alfa:
        return 0
    return 1
    # TODO: Nie trzeba tu dawać else - jeżeli if będzie spełniony to funkcja zakończy swoje działanie w kroku 'return 0'
    #  Natomiast jeżeli warunek nie zostanie spełniony to przejdzie do return 1. Ponadto warto nazywać funckje tak jak
    #  tutaj. "L" mówi Ci coś tylko parę tygodni po napisaniu funkcji a innym deweloperom już w ogóle nic


liczba_l = 0

n = 10
u = 100

for i in range(n):
    x = generator_liczb_losowych(-3, 3, u)
    test_lillieforsa(x)
    liczba_l += test_lillieforsa(x)

liczba_los = 0
for i in range(n):
    x = norm.rvs(size=u)
    test_lillieforsa(x)
    liczba_los += test_lillieforsa(x)
    # TODO: tutaj nie było konieczności nadpisywać zmiennej i. Tak jest dużo czytelniej - wiadomo że i mówi tylko o
    #  indeksie iteracji a x jest zmienną jakiejś funkcji matematycznej.

with open('results_lillieforse.csv', 'a+') as file:
    file.write('Test L dla {} losowan z generatora {} liczb: {}'.format(n, u, liczba_l/n))
    file.write('Test L dla {} losowan z normalnego {} liczb: {}'.format(n, u, liczba_los/n))
# TODO: 'with' gwarantuje, że jak coś się wykrzaczy w bloku pod nim to pliki, zmienne, pamięć itd. zostanie w stanie
#  sprzed tego bloku. Dodatkowo nie musisz nużywać stodout, czyli nie musisz nadpisywać defaultowego struminia zapisu
#  danych
