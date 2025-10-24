from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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

def plot_cross_sections():

    fig, ax = plt.subplots(figsize=(8, 5))

    def cross_section(process):
        E, sig = np.loadtxt(f'data/U-235(N,{process}).csv',
                            skiprows=1, unpack=True, delimiter=',')

        if process=='G':
            l = 'Absorption'
        elif process == 'F':
            l = 'Fission'
        elif process == 'EL':
            l = 'Elastic'
        elif process == 'INL':
            l = 'Inelastic'

        ax.plot(10**E, 10**sig, alpha = 0.9, label=l)
        return E, sig

    # cross_section('TOT')
    cross_section('G')
    cross_section('F')
    cross_section('EL')
    cross_section('INL')
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel(r'$\sigma$ (barn)')
    ax.legend()

    ax.annotate(
        '',
        xy=(1e-6, 1e4),
        xytext=(1e-11, 1e4),
        arrowprops=dict(arrowstyle='<->', lw=1, color='k')
    )
    ax.text(np.sqrt(1e-6*1e-11), 1e4 + 0.05,
            r"$Thermal$", ha='center', va='bottom')

    ax.annotate(
        '',
        xy=(1e-1, 1e4),
        xytext=(1e-6, 1e4),
        arrowprops=dict(arrowstyle='<->', lw=1, color='k')
    )
    ax.text(np.sqrt(1e-1*1e-6), 1e4 + 0.05,
            r"$Epithermal$", ha='center', va='bottom')

    ax.annotate(
        '',
        xy=(30, 1e4),
        xytext=(1e-1, 1e4),
        arrowprops=dict(arrowstyle='<->', lw=1, color='k')
    )
    ax.text(np.sqrt(30*1e-1), 1e4 + 0.05,
            r"$Fast$", ha='center', va='bottom')

    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig('cross_sections.jpg', dpi=300, bbox_inches='tight')
    plt.show()

def plot_energy_distribs():
    
    def energy_sampler(process, number):

        if process=='fast':
            # The distribution is a Watt spectrum. 
            # We define watt spectrum and normalise it so that the sum of probabilities over the grid is 1.
            # Find the Watt CDF which is the probability distribution of watt energies.  
            # Then we invert the CDF so that if we input a uniform random number, the output will be watt energies sampled from the Watt CDF.

            watt_energies = np.linspace(0, 30, 10000)
            watt_spectrum = 0.4865*np.sinh(np.sqrt(2*watt_energies))*np.exp(-watt_energies)
            watt_spectrum /= sum(watt_spectrum)
            watt_cdf = np.cumsum(watt_spectrum)
            watt_cdf /= watt_cdf[-1]
            watt_inverse_cdf = interp1d(watt_cdf, watt_energies)

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

    N = 100000
    bins = 1000

    fig, ax = plt.subplots(1, 3, figsize=(14, 7), tight_layout=True)

    e = energy_sampler('thermal', N)
    x = np.linspace(1e-11, 1e-6, 10000)
    kBT = 0.0253e-6
    ax[0].hist(e, bins=bins, density=True)
    f = 2/np.sqrt(np.pi) * np.sqrt(x) / (kBT**1.5) * np.exp(-x/kBT)
    ax[0].plot(x, f, label = r'$\frac{2}{\sqrt{\pi}(k_BT)^{3/2}}\sqrt{E}\mathrm{e}^{-E/k_BT}$')
    ax[0].set_title('Thermal \n(Maxwell-Boltzmann)')

    e = energy_sampler('epithermal', N)
    x = np.linspace(1e-6, 0.1, 10000)
    ax[1].hist(e, bins=bins, density=True)
    y = 1/x
    y/= np.log(0.1/1e-6)
    ax[1].plot(x, y, label = r'$\frac{1/E}{\mathrm{ln}(E_{max}/E_{min})}$')
    ax[1].set_ylim([0,100])
    ax[1].set_title('Epithermal \n(1/E)')

    e = energy_sampler('fast', N)
    x = np.linspace(0.1, 30, 10000)
    ax[2].hist(e, bins=bins, density=True)
    ax[2].plot(x, 0.4865*np.sinh(np.sqrt(2*x))*np.exp(-x), label = r'$0.4865\sinh(\sqrt{2E})\mathrm{e}^{-E}$')
    ax[2].set_title('Fast \n(Watt)')

    for i in range(3):
        ax[i].set_xlabel('E (MeV)', fontsize=15)
        ax[i].set_ylabel('PDF', fontsize=15)
        ax[i].legend(fontsize=18)

    plt.savefig('probability_distribs.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def plot_scatter():
    fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
    filename = "data/angular_data.txt"
    energies=[]
    all_data = []
    with open(filename, "r") as f:
        data = []
        for line in f:
            if line.startswith('#'):
                energies.append(float(line.strip().split()[1].split('Ei')[1])/1e6)
            else:
                data.append([float(line.strip().split()[0]), float(line.strip().split()[1])])
        
    data = np.array(data)
    all_data = []
    
    start = 0
    for i in range(len(data)-1):
        if data[i][0]<data[i+1][0]:
            all_data.append(data[start:i+1])
            start = i+1

    i = 0
    for data in all_data:
        dsigdomega = data[:, 1]/max(data[:, 1])
        E = energies[i]
        exponent = int(np.floor(np.log10(E)))
        mantissa = E / 10**exponent
        label = rf"${mantissa:.1f}\times 10^{exponent}$"
        if E<0.2:
            ls = '--'
        else:
            ls='-'
        plt.plot(data[:, 0], dsigdomega, label=label, ls=ls, lw=1.5)
        i+=1
    

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Neutron Energy (MeV)')
    plt.ylabel(r'$\mathrm{d}\sigma/\mathrm{d}\Omega$ (barn/sr)')
    plt.xlabel(r'$\theta({}^o)$')
    plt.savefig('angular_distribs.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def plot_results(sets):
    
    fig, ax = plt.subplots(1,2, figsize=(12, 6), tight_layout=True)
    
    axins = inset_axes(ax[1], width="45%", height="45%", loc="lower right", borderpad=1)
    x1, x2 = 0, 100  # x-range to zoom
    y1, y2 = 0.8, 2  # y-range to zoom

    for s in sets:
        df1 = pd.read_csv(f'data/result_{s[0]}_{s[1]}.csv')
        ax[0].plot(df1['Radius'], df1['keff'], label='crude angular dependence')
        ax[1].plot(df1['Mass'], df1['keff'], label=f'crude angular dependence')

        df = pd.read_csv(f'data/result_{s[0]}_{s[1]}_isotropic.csv')
        ax[0].plot(df['Radius'], df['keff'], label='isotropic')
        ax[1].plot(df['Mass'], df['keff'], label=f'isotropic')

        axins.plot(df1['Mass'], df1['keff'])
        axins.plot(df['Mass'], df['keff'])
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

    ax[0].legend(title='100\% Enrichment\nNo reflector', loc='lower right')
    ax[0].set_xlabel('Radius')
    ax[0].set_ylabel(r'$k_{eff}$')

    ax[1].set_xlabel('Mass')
    ax[1].set_ylabel(r'$k_{eff}$')

    ax[0].set_ylim([0.2, 2])
    ax[1].set_ylim([0.2, 2])

    plt.savefig('fig/result 3.jpg', dpi=300, bbox_inches='tight')

# plot_results([[65, 0], [85, 0], [100, 0]])
# plot_results([[65, 70], [85, 70], [100, 70]])

plot_results([[100,0]])

plt.show()