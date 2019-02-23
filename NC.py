import numpy as np
import numpy.linalg as nl
import scipy.integrate as integrate
from mpmath import*
import matplotlib.pyplot as plt

#----Quelques fonctions utiles----
def plus(f,g):
    return (lambda x : f(x)+g(x) )

def times(f,g):
    return (lambda x : f(x)*g(x) )

def RE(f):
    return (lambda x : float(re(f(x))))

def IM(f):
    return (lambda x : float(im(f(x))))    

#----paramètres du problème---- 
L=50
N=2
nd = 7 #nombre de points dans l'espace des k 
K=np.linspace(-pi/L,pi/L,nd)
X0=[(-pi/L - 1/L) for k in K] 
Vper = lambda x : sin(2*pi*x/L) #le potentiel 

#--------------------------------------------------------------------

def e(n): #la base de discrétisation
    return (lambda x : (1/sqrt(L))*exp(j*2*n*pi*x/L) )

def de(n):  #la dérivée des vecteurs de la base
    return (lambda x : (j*2*n*pi/(L*sqrt(L)))*exp(j*2*n*pi*x/L) )

#--------------------------------------------------------------------

def a(k,m,n): #calcule a_k(em,en)
    T1 = integrate.quad(  RE( times( de(m),de(n) ) ) , -L/2 , L/2 )[0] + j*integrate.quad( IM( times( de(m),de(n) ) ) , -L/2 , L/2 )[0]
    T2 = - (2*j*k)*( integrate.quad( RE(times(de(m),e(n))) , -L/2 , L/2 )[0] + j*integrate.quad( IM(times(de(m),e(n))) , -L/2 , L/2 )[0] )
    T3 = (k**2)*(integrate.quad( RE(times(e(m),e(n))), -L/2 , L/2 )[0]+j*integrate.quad( IM(times(e(m),e(n))), -L/2 , L/2 )[0])
    T4 = integrate.quad(RE(times(times(Vper,e(m)),e(n))), -L/2 , L/2 )[0] +j*integrate.quad(IM(times(times(Vper,e(m)),e(n))), -L/2 , L/2 )[0]
    T = T1 + T2 + T3 + T4
    return(T)

def b(m,n):  #calcule b(em,en)
    B = integrate.quad ( RE(times(e(m),e(n))), -L/2 , L/2 )[0] + j*integrate.quad ( IM(times(e(m),e(n))), -L/2 , L/2 )[0]
    return(B)

#--------------------------------------------------------------------

def B(): #la matrice B
    B1=np.zeros( (2*N+1,2*N+1),dtype=complex )
    for m in range(2*N+1):
        for n in range(2*N+1):
            B1[m,n]=b(n-N,m-N)
    return(B1)

def A(k): #la matrice A_k
    A1=np.zeros( (2*N+1,2*N+1),dtype=complex )
    for m in range(2*N+1):
        for n in range(2*N+1):
            A1[m,n]=a(k,n-N,m-N)
    return(A1)

def S(k):  #la matrice S_k
    return( np.dot(nl.inv(B()),A(k)) )

#-------------------------------------------------------------------

def VP(k): #les valeurs propres de S_k
    D, U = nl.eig(S(k))
    D.sort()
    return (D)

#----résolution et affichage----

Y=np.zeros((2*N+1,len(K)))
for i in range(len(K)) :
    V=VP(K[i])
    for n in range(2*N+1):
        Y[n,i]=re(V[n])

for n in range(2*N+1):
    plt.plot(K,Y[n,:])
    plt.plot(X0,Y[n,:])

plt.xlabel('k')
plt.ylabel('E_k,n')

plt.show()
