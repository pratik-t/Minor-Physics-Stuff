import numpy as np
import itertools as it
from fractions import Fraction as frac

def equivalent(n,l):

    def comb(x):
        temp= [list(i) for i in it.permutations(x)]
        l = []
        [l.append(i) for i in temp if i not in l]
        return l

    def reduce(tab,i,j,r,c):
        for x in range(i,r):
            for y in range(j,c):
                if tab[x,y]!=0: tab[x,y]-=1
        return tab

    N= n

    if n>2*l+1: n= 2*(2*l+1)-n

    Lsym=['S','P','D','F','G','H','I','J','K','L','M','N']

    x= [i for i in range(-l,l+1)]
    ms= np.linspace(-n*0.5,n*0.5, num= n+1)
    slater=[]

    base= [1,2]
    if l>0: [base.append(int(i)) for i in np.zeros((2*l+1)-2)]
    else: base=[1,0]

    temp=[list(i) for i in it.combinations_with_replacement(base,2*l+1)]
    y=[]
    [y.append(i) for i in temp if sum(i)==n and i not in y]

    table=[]
    for i in y: [table.append(j) for j in comb(i)]
    [table.remove(i) for i in table if np.dot(i,x)<0]

    for ML in range(n*l,-1,-1):
        arrangement= []
        [arrangement.append(i) for i in table if np.dot(i,x)==ML]

        slat=[]

        for i in arrangement:
            spins= list(it.product([-0.5,0.5],repeat=i.count(1)))
            s= [sum(j) for j in spins]
            count=[]
            for k in ms:
                count.append(s.count(k))
            if slat==[]:
                slat=count
            else:
                slat=[slat[i]+count[i] for i in range(len(count))]

        if slat!=[]: slat.insert(0,ML); slater.append(slat)

    if np.array(slater).size==0:
        print('\nConfguration not allowed')
        exit()

    print('\nSLATER TABLE IS:\n')
    print('ML/MS', end='')
    for i in ms: print(f'\t{str(frac(i))}', end='')
    print('\n')
    for i in slater:
        for j in i:
            print(f' {j}', end= '\t')
        print('\n')

    print('ATOMIC TERM SYMBOLS ARE (Increasing order of energy- Hund\'s Rule):\n')

    slater=np.array(slater)
    tab= slater[:,1:]

    r= np.shape(tab)[0]
    c= np.shape(tab)[1]

    terms= []

    for i in range(r):
        for j in range(c):
            if tab[i,j]!=0:
                while tab[i,j]!=0:
                    L=slater[i,0]
                    s= abs(ms[j])
                    S= int(2*s+1)
                    terms.append([S,L])

                    tab= reduce(tab,i,j,r,c)

    terms.sort(reverse=True)

    for i in terms:
        S= i[0]
        s= (1/2)*(S-1)
        L= i[1]
        J=np.arange(abs(L-s),L+s+1,1)

        if N>2*l+1: J[::-1].sort()

        print(f'{S}{Lsym[L]} ( ',end='')
        for x in J: print(f'{str(frac(x))}. ',end='')
        print(')\n')

linput = str(input("\nEnter l (s,p,d,f): "))

lsym=['s','p','d','f','g','h']
try:
    l=lsym.index(linput)
except:
    try:
        l = int(linput)
        print(f'> l is {lsym[l]}\n')
    except: 
        print('Enter valid l')
        exit()

n = int(input("Enter no. of equivalent electrons: "))


equivalent(n,l)