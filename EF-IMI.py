import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nl
from math import *

L = 10000
N = 200
nk = 100
K=np.linspace(-pi/L,pi/L,nk)
X0=[(-pi/L - 1/L) for k in K]
Vper = lambda x : 5*sin(x);
#------------------------------------------------------------------------------
def Fourier(V,N,L):
    vi = np.array([ V(n*L/(2*N+1)) for n in range(2*N+1) ])
    vf1 = np.fft.fft(vi)
    #vf2 =  np.fft.fftshift(vf1) /(2*N+1)
    vf2 = vf1 / (2*N+1)
    return(vf2)
    

def A(k,V,N):
    VF=Fourier(V,2*N,L)
    A1=np.zeros( (2*N+1,2*N+1),dtype=complex )
    for m in range(2*N+1):
        for n in range(2*N+1):
            if ( m == n ):
                A1[m,n] = (2*pi*(n-N)/L)**2 + k*4*pi*(n-N)/L + k**2 + VF[0]
            else :
                A1[m,n] = VF[m-n]
    return(A1)


def VP(k,V,N): #les valeurs propres de A_k
    D, U = nl.eig(A(k,V,N))
    D.sort()
    return (D)
#----------------------------------------------------------------------------
Y=np.zeros((2*N+1,len(K)))
for i in range(len(K)) :
    V=VP(K[i],Vper,N)
    for n in range(2*N+1):
        Y[n,i]=V[n].real

for n in range(2*N+1):
    plt.plot(K,Y[n,:])
    

plt.xlabel('k')
plt.ylabel('E_k,n')

plt.show()
