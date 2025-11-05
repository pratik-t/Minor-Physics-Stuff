import numpy as np


enrichment = [65, 85, 100]
reflection = [0, 70]

for r in reflection:
    for e in enrichment:
        fn = f'data/result_{e}_{r}.csv'

        ra, m, keff = np.loadtxt(fn, usecols=[1,2,3], unpack=True, delimiter=',', skiprows=1)
        mk1 = np.interp(1, keff, m)
        rak1 = np.interp(1, keff, ra)
        print(f'Enrichment: {e} Reflection: {r} M(k_eff=1): {mk1:.3} kg R(k_eff=1): {rak1:.3} cm')
