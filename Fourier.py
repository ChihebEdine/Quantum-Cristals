import numpy as np
from mpmath import*
import matplotlib.pyplot as plt


def Fourier(f,T,N,i):
    #f de période T
    #donne le iéme coéfficient de fourrier de f pour  i allant de -N à N 
    Li=np.array([ complex(f(n*T/(2*N+1))) for n in range(2*N+1) ])
    Lf=np.fft.fft(Li)
    if (i>=-N and i<=N):
        return(Lf[i]/(2*N+1))
    else :
        print('out of range, -N<=i<=N !!')


def Fourier_inv(F):
    #F liste des coéfficients d'indices allant de 0 à N puis de -N à -1 de Fourier d'une fonction f
    #f de periode T
    #retourne la liste des valeurs de f en kT/(len(F)
    Li=np.array(F)
    Lf=np.fft.ifft(len(F)*Li)
    return(Lf)

def PLOT_r(iF,T):
    X=[ k*T/(len(iF))    for k in range(len(iF)) ]
    Y=[ float(re(iF[k])) for k in range(len(iF)) ]
    plt.plot(X,Y)
    plt.show()




