from matplotlib.ticker import MaxNLocator
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import multiprocessing as mp
from tqdm import tqdm

BARN_TO_CM2 = 1e-24
AVOGADRO = 6.0221408e+23

CROSS_SECTIONS = {}
for process in ['TOT', 'F', 'EL', 'INL', 'G']:
    logE, logsig = np.loadtxt(f'data/U-235(N,{process}).csv', skiprows=1, unpack=True, delimiter=',')
    CROSS_SECTIONS[process] = (logE, logsig)

watt_energies = np.linspace(0, 30, 10000)
watt_spectrum = 0.4865*np.sinh(np.sqrt(2*watt_energies))*np.exp(-watt_energies)
watt_spectrum /= sum(watt_spectrum)
watt_cdf = np.cumsum(watt_spectrum)
watt_cdf /= watt_cdf[-1]
watt_inverse_cdf = interp1d(watt_cdf, watt_energies)


def energy_sampler(process, number):

    if process=='fast':
        # The distribution is a Watt spectrum.
        # We define watt spectrum and normalise it so that the sum of probabilities over the grid is 1.
        # Find the Watt CDF which is the probability distribution of watt energies.
        # Then we invert the CDF so that if we input a uniform random number, the output will be watt energies sampled from the Watt CDF.
        # TBH this is pretty close to Maxwellian

        return watt_inverse_cdf(np.random.rand(number))

    elif process=='epithermal':

        # This is the resonance region. Distribution is 1/E
        # CDF is integral(1/E)_E_min^E = ln(E/E_min)
        # Normalised CDF is ln(E/E_min)/ln(E_max/E_min)
        # if CDF(E) = uniform then can invert: E = Emin​(Emax​/Emin​)^uniform

        # Energy range is 10^-6 MeV to 0.1 MeV
        E_min = 1e-6
        E_max = 0.1
        r = np.random.uniform(size=number)
        return E_min * (E_max / E_min) ** r

    elif process=='thermal':

        # This is the maxwell botzmann distribution
        # CDF is chi^2 distribution with degree of freedom = 3
        # Scale by <E>= 1/2 k_B*T (for our cross sections, T=293K so kBT = 0.0253*10^-6 MeV) to get energy in MeV
        # Actually <E> is 3/2 kT but <chi^2> with dof 3 is 3.
        # Energy range is 10^-11 MeV to 10^-6 MeV
        kBT = 0.0253e-6
        return np.random.chisquare(df=3, size= number) * 0.5 * kBT


def graph_interactions(radius, enrichment, reflection, max_gen, burn_in, min_tail, save=False):

    results = run(enrichment, reflection, radius, 1,
              max_gen, 100, all_neutrons_flag=True)

    masses = results[0]
    k_vals = results[1]
    all_neutrons = results[2]

    c = plt.get_cmap('tab20b').colors

    plt.rcParams.update({
        "text.usetex": True,                 # Use LaTeX for all text
        "font.family": "serif",              # Use serif fonts (like LaTeX default)
        "font.size": 16,                     # Default font size
        "axes.labelsize": 18,                # Axis labels
        "axes.titlesize": 20,                # Title size
        "xtick.labelsize": 18,               # X tick labels
        "ytick.labelsize": 18,               # Y tick labels
        "legend.fontsize": 18,               # Legend text
        "lines.markersize": 8,               # Marker size
        "lines.linewidth": 1.2,                # Line width
        "xtick.major.size": 6,               # Major tick length
        "ytick.major.size": 6,
        "xtick.major.width": 1.2,            # Major tick width
        "ytick.major.width": 1.2,
        "xtick.direction": "in",             # Point ticks inward
        "ytick.direction": "in",
        "axes.grid": True,                   # Enable grid
        "grid.alpha": 0.5,                   # Grid transparency
        "grid.linestyle": "--",              # Dashed grid lines
        "grid.linewidth": 0.8,
    })

    fig, ax = plt.subplots(1, 2, figsize=(12,6), gridspec_kw={'width_ratios': [3, 1]}, tight_layout=True)
    for n in all_neutrons:
        x = np.linspace(n.gen, n.gen + 1, len(n.radial_trajectory))
        ax[0].plot(x, n.radial_trajectory, c=c[n.gen % len(c)])
        ax[0].axhline(radius, ls=':', c='k', lw=2, alpha=0.5)

    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_ylim([-radius/10, radius+radius/10])
    ax[0].set_ylabel('Radius (cm)')
    ax[0].set_xlabel('Generation')
    ax[0].set_title('Neutron interaction chains')

    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel(r'$k_{eff}$')

    x = np.linspace(1, len(k_vals), len(k_vals))

    if len(k_vals) >= burn_in+min_tail:
            ax[1].plot(x[:10], k_vals[:10], ':', c='tab:blue')
            ax[1].plot(x[9:], k_vals[9:], '-', c='tab:blue')
    else:
        ax[1].plot(x, k_vals, '-', c='tab:blue')

    ax[1].axhline(np.mean(k_vals), c='tab:red', label='Mean')
    ax[1].legend(loc='lower right')
    
    if save:
        plt.savefig('neutron_intreractions.jpg', dpi=300, bbox_inches='tight')
    
    plt.show()


class Neutron():
    def __init__(self, gen, position, logenergy):
        self.gen = gen
        self.position = position
        self.logenergy = logenergy #log10(energy)
        self.interactions = 0
        self.radial_trajectory = [np.linalg.norm(position)]

    def trajectory(self, xf, xf_proc):
        
        # if xf_proc in ['INL', 'F', 'G']:
        #     cos_theta = np.random.uniform(-1, 1) #angles upto 25deg
        # elif xf_proc == 'EL':
        #     if 10**self.logenergy<5e-2: 
        #         cos_theta = np.random.uniform(-1.0, 1.0) # all angles
        #     elif 10**self.logenergy<1e-1: 
        #         cos_theta = np.random.uniform(np.cos(np.deg2rad(130)), np.cos(np.deg2rad(0))) # angles upto 130deg
        #     else:
        #         cos_theta = np.random.uniform(np.cos(np.deg2rad(50)), np.cos(np.deg2rad(0))) #angles upto 50deg
        
        cos_theta = np.random.uniform(-1.0, 1.0)

        phi = np.random.uniform(0.0, 2*np.pi)
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))
        unit_dir =  np.array([sin_theta*np.cos(phi),
                        sin_theta*np.sin(phi),
                        cos_theta])

        self.position = self.position + xf * unit_dir
        
        rad = np.linalg.norm(self.position)
        
        self.radial_trajectory.append(rad)

        return rad


    def mean_free_path(self, radius, number_density, reflection):

        processes = ['F', 'EL', 'INL', 'G']
        n_proc = 4

        fission_products = []
        
        logE_arrays = [CROSS_SECTIONS[p][0] for p in processes]   # arrays of log10(E)
        logsig_arrays = [CROSS_SECTIONS[p][1] for p in processes]
        
        interactions = 0
        
        while True:

            interactions+=1
            logsigs = np.array([np.interp(self.logenergy, logE_arrays[i], logsig_arrays[i]) for i in range(n_proc)])
            sigs = 10**logsigs
            rnd = np.random.uniform(0, 1, size = n_proc)
            
            with np.errstate(divide='ignore', invalid='ignore'): # avoid warnings
                xfs = -np.log(rnd) / (BARN_TO_CM2 * sigs * number_density)
            
            for i in range(n_proc): # set xf to infinity if energy beyond interpolation range
                logE = logE_arrays[i]
                if (self.logenergy < logE[0]) or (self.logenergy > logE[-1]):
                    xfs[i] = np.inf

            idx = np.argmin(xfs)
            xf_value = xfs[idx]
            xf_proc = processes[idx]

            # r2 = self.position**2 + xf_value**2 + 2 * self.position * xf_value * angle
            # new_pos = np.sqrt(r2) if r2 > 0 else 0.0
            # new_pos = self.position**2 + xf_value**2 + 2 * self.position * xf_value * angle

            new_pos = self.trajectory(xf_value, xf_proc)

            if (interactions >= 500):
                return []
            
            if (new_pos > radius):
                # print(f'gen {self.gen} → neutron lost')
                if np.random.uniform(0,1) >= reflection/100:
                    return []
         

            if xf_proc == 'EL':
                # print(f'gen {self.gen} {xf_proc} → Elastic')
                # elastic scatter: neutron continues with same energy but at new position
                # loop continues (no recursion)
                continue

            elif xf_proc == 'INL':
                # print(f'gen {self.gen} {xf_proc} → Inelastic')
                # inelastic scatter -> produce ONE scattered neutron with lower energy
                # choose thermal with p=0.5 else epithermal
                if np.random.rand() < 0.5:
                    sampled = energy_sampler('thermal', 1)[0]
                else:
                    sampled = energy_sampler('epithermal', 1)[0]

                # child = Neutron(self.gen + 1, self.position, np.log10(sampled))
                # fission_products.append(child)

                # return fission_products
                self.logenergy = np.log10(sampled)
                continue
            
            elif xf_proc == 'G':
                # print(f'gen {self.gen} {xf_proc} → neutron absorbed')
                # inelastic or gamma → neutron dies, so stop here
                return []
            
            elif xf_proc == 'F':
                
                # fission: spawn 2/3 new neutrons (multiplicity) with average of 2.5
                new_neutrons = 3 if random.random() < 0.5 else 2

                # print(f'gen {self.gen} {xf_proc} → {new_neutrons} new neutrons at {self.position}')

                new_energies = energy_sampler('fast', new_neutrons)
                
                for energy in new_energies:
                    fission_products.append(Neutron(self.gen + 1, self.position, np.log10(energy)))
                    
                return fission_products

            else:
                print('Unknown Process')
                return []


def run(enrichment, reflection, radius, starting_neutrons, max_gen, MAX_NEUTRONS=100, all_neutrons_flag=False):
    
    # energies can always be left in log10 as they are only used to interpolate cross section.
    # Except when you need to sample an energy itself, say after fission. Then sample from distribution, then convert to log10 for interpolation.

    volume = 4*np.pi*radius**3/3
    mass_density = 19.05  # (g/cm³)
    mass = mass_density*volume
    number_of_atoms = mass * AVOGADRO/235.044  # molar mass = 235.44 g/mol
    number_density = (number_of_atoms/volume)*(enrichment/100)
    
    gen = 0

    positions = [0,0,0]
    e1 = np.log10(energy_sampler('thermal', int(starting_neutrons/2))) # start with therm+epi distributed energies
    e2 = np.log10(energy_sampler('epithermal', int(starting_neutrons/2)))
    energies = np.concatenate((e1, e2))
    
    k_values = []
    all_neutrons = []
    current_gen = [Neutron(gen, positions, energies[i]) for i in range(starting_neutrons)]

    while (gen < max_gen) and len(current_gen) > 0:
        
        # print(f'gen {gen} N: {len(current_gen)}')

        if all_neutrons_flag:
            all_neutrons.extend(current_gen)
        
        next_gen = []
        for n in current_gen:
            next_gen += n.mean_free_path(radius, number_density, reflection)

        raw_k = len(next_gen) / max(1, len(current_gen))
        k_values.append(raw_k)

        if len(next_gen) == 0:
            # extinction: stop the run
            current_gen = []
        else:
            if len(next_gen) >= MAX_NEUTRONS:
                # sample without replacement to get the next simulated generation
                current_gen = random.sample(next_gen, MAX_NEUTRONS)
            else:
                # too few neutrons in bank
                current_gen = next_gen

        gen+=1

    return mass/1000, k_values, all_neutrons
    

def run_with_args(args):
    return run(*args)


def worker_init():
    np.random.seed()
    random.seed()


def critical(N_workers, N_runs, radius, starting_neutrons, max_gen, enrichment, reflection):

    net_k = 0
    net_len = 0

    all_k_vals = []

    args_list = [(enrichment, reflection, radius, starting_neutrons, max_gen, 100) for _ in range(N_runs)]
    
    results = []
    with mp.Pool(processes=N_workers, initializer=worker_init) as pool:
        for r in tqdm(pool.imap(run_with_args, args_list), total=N_runs):
            results.append(r)

    masses, k_lists, all_neutrons = zip(*results)

    mass = masses[0]

    burn_in = 10 # number of generations after which to start averaging
    min_tail = 10 # minimum number of generations after burn-in to start averaging
    
    for k_vals in k_lists:
        
        if len(k_vals) >= burn_in + min_tail:
            net_k += np.sum(k_vals[burn_in:])
            net_len += len(k_vals[burn_in:])
        else:
            net_k += np.sum(k_vals)
            net_len += len(k_vals)

        # x = np.linspace(1, len(k_vals), len(k_vals))
        
        # if len(k_vals)>=burn_in+min_tail:
        #     plt.plot(x[:10], k_vals[:10], ':', c='tab:blue', alpha=0.5)
        #     plt.plot(x[9:], k_vals[9:], '-', c='tab:blue', alpha=0.5)
        # else:
        #     plt.plot(x, k_vals, '-', c='tab:blue', alpha=0.5)

    k_eff = net_k/net_len

    return radius, mass, k_eff

# radius = 10
# starting_neutrons = 1
# enrichment = 85
# reflection = 0
# max_gen = 30

# burn_in = 10  # number of generations after which to start averaging
# min_tail = 10  # minimum number of generations after burn-in to start averaging
# graph_interactions(radius, enrichment, reflection, max_gen, burn_in, min_tail, save=True)
# plt.show()