import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation, FuncAnimation, PillowWriter
import matplotlib.colors as col
from itertools import cycle

import warnings
warnings.filterwarnings("ignore")

def legendre(l, x):

    if l==0:
        P= 1
    elif l==1:
        P= x
    else:
        P= (1/l)*((2*l-1)*x*legendre(l-1,x)-(l-1)*legendre(l-2,x))

    return P

def alegendre(l, m, x):

    x= np.array(x)

    if m==0:
        Plm= legendre(l,x)
    elif m>0:
        A= 1/np.sqrt(1-x**2)
        A[A==np.inf]= 0
        Plm= A*((l-m+1)*x*alegendre(l,m-1,x)- (l+m-1)*alegendre(l-1,m-1,x))
    else:
        m= abs(m); Plm= (-1)**m*(math.factorial(l-m)/math.factorial(l+m))*alegendre(l,m,x)

    return Plm

def harmonics(l, m, dn):

    A= np.sqrt(((2*l+1)/(4*np.pi))*(math.factorial(l-abs(m))/math.factorial(l+abs(m))))

    phi= np.linspace(0,2*np.pi,dn)
    theta= np.linspace(0,np.pi,dn)
    theta, phi= np.meshgrid(theta, phi)

    if m<0:
        Ylm= A*alegendre(l,m,np.cos(theta))*np.sin(abs(m)*phi)
    else:
        Ylm= A*alegendre(l,m,np.cos(theta))*np.cos(m*phi)

    return Ylm

def radial_graph(l, m, fig):

    def plot(r,dn, fig):

        phi= np.linspace(0,2*np.pi,dn)
        theta= np.linspace(0,np.pi,dn)
        theta, phi= np.meshgrid(theta, phi)

        x= abs(r)*np.cos(phi)*np.sin(theta)
        y= abs(r)*np.sin(phi)*np.sin(theta)
        z= abs(r)*np.cos(theta)
        lim= r.max()

        ax = fig.add_subplot(1,2,1, projection='3d')
        ax.plot_surface(x,y,z,cmap='viridis')
        ax.axis('off')
        ax.set_box_aspect((1,1,1))
        ax.axes.set_xlim3d(left=-lim, right=lim)
        ax.axes.set_ylim3d(bottom=-lim, top=lim)
        ax.axes.set_zlim3d(bottom=-lim, top=lim)

    waves= harmonics(l,m,100)
    plot(waves,100,fig)

    return


def animation1(dn, frn, r, theta, phi, ax, newcmp):

        x= np.zeros((dn, dn, frn))
        y= np.zeros((dn, dn, frn))
        z= np.zeros((dn, dn, frn))

        for i in range(frn):
            x[:,:,i]= abs(r[:,:,i])*np.cos(phi)*np.sin(theta)
            y[:,:,i]= abs(r[:,:,i])*np.sin(phi)*np.sin(theta)
            z[:,:,i]= abs(r[:,:,i])*np.cos(theta)


        def change_plot(frame,r, plot):
            plot[0].remove()
            facecol = newcmp((r[:,:,frame]))

            plot[0]= ax.plot_surface(x[:,:,frame],y[:,:,frame],z[:,:,frame],
                                        facecolors=facecol)


        plot = [ax.plot_surface(x[:,:,0], y[:,:,0], z[:,:,0],
                    facecolors = newcmp((r[:,:,0])))]

        ani = FuncAnimation(fig, change_plot, frn, fargs=(r, plot), interval=1, repeat= False)
        
        f = r"./animation.gif" 
        writergif = PillowWriter(fps=10) 
        ani.save(f, writer=writergif)
        plt.show()

def animation2(frn, r, theta, phi, ax, newcmp):

    ims= []
    for i in range(frn):
        x= abs(r[:,:,i])*np.cos(phi)*np.sin(theta)
        y= abs(r[:,:,i])*np.sin(phi)*np.sin(theta)
        z= abs(r[:,:,i])*np.cos(theta)

        facecol = newcmp((r[:,:,i]))
        sf = ax.plot_surface(x, y, z, facecolors=facecol)

        lim= 1.5
        ax.axis('off')
        ax.set_box_aspect((1,1,1))
        ax.axes.set_xlim3d(left=-lim, right=lim)
        ax.axes.set_ylim3d(bottom=-lim, top=lim)
        ax.axes.set_zlim3d(bottom=-lim, top=lim)

        ims.append([sf])

    ims= cycle(ims)
    flag= 0
    new_ims= []
    for i in ims:
        new_ims.append(i)
        flag+=1
        if flag>500: break


    ani= ArtistAnimation(fig, new_ims, interval= 50, blit= True, repeat= False)
    f = r"./animation.gif"
    writergif = PillowWriter(fps=10) 
    ani.save(f, writer=writergif)
    plt.show()


def vibrations(l, m, dn, fig, use_anim):

    if l<3: r_osc= 0.5
    elif l<7: r_osc= 0.5
    elif l<15: r_osc= 0.3
    else: r_osc=0.2

    if use_anim==1: frn= 12
    else: frn= 30

    phi= np.linspace(0,2*np.pi,dn)
    theta= np.linspace(0,np.pi,dn)
    theta, phi= np.meshgrid(theta, phi, sparse=True)

    waves= harmonics(l,m,dn)

    R= np.ones((dn,dn))

    r1= np.linspace(R-r_osc*(waves),R+r_osc*(waves),int(frn/2))
    r2= np.linspace(R+r_osc*(waves),R-r_osc*(waves),int(frn/2))
    rnet= np.concatenate((r1,r2))

    r= np.zeros((dn, dn, frn))

    for element in range(frn):
        r[:,:,element]= rnet[element]

    ax = fig.add_subplot(122, projection='3d')
    lim= 1.5
    ax.axis('off')
    ax.set_box_aspect((1,1,1))
    ax.axes.set_xlim3d(left=-lim, right=lim)
    ax.axes.set_ylim3d(bottom=-lim, top=lim)
    ax.axes.set_zlim3d(bottom=-lim, top=lim)
    viridisBig = plt.cm.get_cmap('viridis', 512)
    newcmp = col.ListedColormap(viridisBig(np.linspace(2.5, 0.4, 512)))
    # norm= col.Normalize()


    if use_anim==1: animation1(dn, frn, r,theta,phi,ax, newcmp)
    else: animation2(frn, r,theta,phi,ax, newcmp)




l= int(input("Enter l: "))
m= int(input("Enter m: "))
anim_type = int(input('Use animation type? (1 or 2): '))
# anim_type = 1

# The first type of animation using FUNCANIMATION: animate1()
# This cannot use blitting, so the animation is slow. But the interactivity is
# good. Make dn smaller to use this animation. Start time is FAST
# The second type of animation using ARTISTANIMATION: animate2()
# This first makes an array of surface plots and then blits through it making it
# quite smooth and fast even with large dn. Interactivity is horrible and laggy.
# Start time is SLOW


fig = plt.figure()

plt.suptitle(f'Spherical Harmonics: Radial Plot and Oscillations on Sphere\n\n\
                l={l}, m={m}')

radial_graph(l,m, fig)

if anim_type==1: segments= 500
else: segments= 50

vibrations(l,m,segments,fig, anim_type)
