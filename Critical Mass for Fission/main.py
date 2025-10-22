from fission import critical
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd

N_workers = 45
N_runs = 500

starting_neutrons = 100
max_gen = 30
enrichment = [100]
reflection = [0]

radii = np.linspace(0.1, 25, 100)


for e in enrichment:
    for r in reflection:

        ench, rad, mass, keff = np.zeros(len(radii)), np.zeros(len(radii)), np.zeros(len(radii)), np.zeros(len(radii))

        filename = f'data/result_{e}_{r}_isotropic.csv'

        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write('Enrichment,Radius,Mass,keff\n')

        for i in range(len(radii)):
            rad, m, k = critical(N_workers, N_runs, radii[i], starting_neutrons, max_gen, e, r)
            
            with open(filename, 'a') as f:
                f.write(f"{e},{rad},{m},{k}\n")

            print(f"Saved: radius={rad:.2f} cm, k_eff={k:.3f}")

# N_workers = 16
# N_runs = 500

# starting_neutrons = 100
# max_gen = 30
# enrichment = 85
# reflection = 0

# radii = [8.5]
# ench, rad, mass, keff = np.zeros(len(radii)), np.zeros(len(radii)), np.zeros(len(radii)), np.zeros(len(radii))

# for i in range(len(radii)):
#     r, m, k = critical(N_workers, N_runs, radii[i], starting_neutrons, max_gen, enrichment, reflection)
#     ench[i] = enrichment
#     rad[i] = r
#     mass[i] = m
#     keff[i] = k
    

#     print(f"Saved: radius={r:.2f} cm, k_eff={k:.3f}")




# data = pd.read_csv('result.csv')

# plt.plot(data['Mass'], data['keff'])
# plt.show()