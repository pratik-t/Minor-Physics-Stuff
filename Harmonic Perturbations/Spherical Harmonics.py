import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation, FuncAnimation, PillowWriter
import matplotlib.colors as col
from itertools import cycle

import warnings
warnings.filterwarnings("ignore")


def legendre(l, x):

    if l == 0:
        P = 1
    elif l == 1:
        P = x
    else:
        P = (1/l)*((2*l-1)*x*legendre(l-1, x)-(l-1)*legendre(l-2, x))

    return P


def alegendre(l, m, x):

    x = np.array(x)

    if m == 0:
        Plm = legendre(l, x)
    elif m > 0:
        A = 1/np.sqrt(1-x**2)
        A[A == np.inf] = 0
        Plm = A*((l-m+1)*x*alegendre(l, m-1, x) -
                 (l+m-1)*alegendre(l-1, m-1, x))
    else:
        m = abs(m)
        Plm = (-1)**m*(math.factorial(l-m) /
                       math.factorial(l+m))*alegendre(l, m, x)

    return Plm


def harmonics(l, m, dn):

    A = np.sqrt(((2*l+1)/(4*np.pi)) *
                (math.factorial(l-abs(m))/math.factorial(l+abs(m))))

    phi = np.linspace(0, 2*np.pi, dn)
    theta = np.linspace(0, np.pi, dn)
    theta, phi = np.meshgrid(theta, phi)

    if m < 0:
        Ylm = A*alegendre(l, m, np.cos(theta))*np.sin(abs(m)*phi)
    else:
        Ylm = A*alegendre(l, m, np.cos(theta))*np.cos(m*phi)

    return Ylm


def radial_graph(l, m, fig):

    def plot(r, dn, fig):

        phi = np.linspace(0, 2*np.pi, dn)
        theta = np.linspace(0, np.pi, dn)
        theta, phi = np.meshgrid(theta, phi)

        x = abs(r)*np.cos(phi)*np.sin(theta)
        y = abs(r)*np.sin(phi)*np.sin(theta)
        z = abs(r)*np.cos(theta)
        lim = r.max()

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis')
        ax.axis('off')
        ax.set_box_aspect((1, 1, 1))
        ax.axes.set_xlim3d(left=-lim, right=lim)
        ax.axes.set_ylim3d(bottom=-lim, top=lim)
        ax.axes.set_zlim3d(bottom=-lim, top=lim)

    waves = harmonics(l, m, 100)
    plot(waves, 100, fig)

    return


def animation(dn, frn, r, theta, phi, ax, newcmp, l, m, save):

    x = np.zeros((dn, dn, frn))
    y = np.zeros((dn, dn, frn))
    z = np.zeros((dn, dn, frn))

    for i in range(frn):
        x[:, :, i] = abs(r[:, :, i])*np.cos(phi)*np.sin(theta)
        y[:, :, i] = abs(r[:, :, i])*np.sin(phi)*np.sin(theta)
        z[:, :, i] = abs(r[:, :, i])*np.cos(theta)

    def change_plot(frame, r, plot):
        plot[0].remove()
        facecol = newcmp((r[:, :, frame]))

        plot[0] = ax.plot_surface(x[:, :, frame], y[:, :, frame], z[:, :, frame],
                                  facecolors=facecol)

    plot = [ax.plot_surface(x[:, :, 0], y[:, :, 0], z[:, :, 0],
                            facecolors=newcmp((r[:, :, 0])))]

    ani = FuncAnimation(fig, change_plot, frn, fargs=(
        r, plot), interval=1, repeat=not save)

    if save:
        f = f"./animation_Y_{l}_{m}.gif"
        writergif = PillowWriter(fps=20)
        ani.save(f, writer=writergif)
    else:
        plt.show()


def vibrations(l, m, dn, fig, frame_rate, save):

    if l < 3:
        r_osc = 0.5
    elif l < 7:
        r_osc = 0.5
    elif l < 15:
        r_osc = 0.3
    else:
        r_osc = 0.2

    frn = frame_rate

    phi = np.linspace(0, 2*np.pi, dn)
    theta = np.linspace(0, np.pi, dn)
    theta, phi = np.meshgrid(theta, phi, sparse=True)

    waves = harmonics(l, m, dn)

    R = np.ones((dn, dn))

    r1 = np.linspace(R-r_osc*(waves), R+r_osc*(waves), int(frn/2))
    r2 = np.linspace(R+r_osc*(waves), R-r_osc*(waves), int(frn/2))
    rnet = np.concatenate((r1, r2))

    r = np.zeros((dn, dn, frn))

    for element in range(frn):
        r[:, :, element] = rnet[element]

    ax = fig.add_subplot(122, projection='3d')
    lim = 1.5
    ax.axis('off')
    ax.set_box_aspect((1, 1, 1))
    ax.axes.set_xlim3d(left=-lim, right=lim)
    ax.axes.set_ylim3d(bottom=-lim, top=lim)
    ax.axes.set_zlim3d(bottom=-lim, top=lim)
    viridisBig = plt.cm.get_cmap('viridis', 512)
    newcmp = col.ListedColormap(viridisBig(np.linspace(2.5, 0.4, 512)))

    animation(dn, frn, r, theta, phi, ax, newcmp, l, m, save)


l = int(input("\nEnter l: "))
m = int(input("Enter m: "))

if (m > l):
    print("\n'm' value must be smaller than 'l' value")
    exit()

s = int(input('Save animation (1) or view interactive (0): '))
save = (s == 1)

fig = plt.figure()

plt.suptitle(f'Spherical Harmonics: Radial Plot and Oscillations on Sphere\n\n\
                l={l}, m={m}')

radial_graph(l, m, fig)

if save:
    segments = 500
    framerate = 30
else:
    segments = 25
    framerate = 12

vibrations(l, m, segments, fig, framerate, save)
