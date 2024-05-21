import numpy as np

import matplotlib.pyplot as plt

# First we'll run a suite of Poisson problems for different n and p:
p_list = [4, 6, 8, 10, 12, 14, 18, 22, 30]

for p in p_list:
    n_list = list(range(2*(p-2), 100, p-2))

    for n in n_list:
        print(n)