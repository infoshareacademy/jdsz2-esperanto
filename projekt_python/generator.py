import numpy as np
import matplotlib.pyplot as plt

def generator_liczb_losowych(xmin, xmax, funkcja_rozkladu):
    lista = funkcja_rozkladu(size=1000)
    losowa = None
    r = np.random.uniform()
    for i in lista:
        if i in [xmin, xmax] and r < i/lista.max:
            losowa = i
            break
    return losowa

if __name__ == '__main__':
    wynik = generator_liczb_losowych(1, 5, np.random.normal)
    print(wynik)
