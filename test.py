from time import time
from math import *

a = range(1, 25000000)


def f(x):
    return 3 * log(x) + cos(x) ** 2


st = time()
r = [f(x) for x in a]
print(f"{time()-st}")
