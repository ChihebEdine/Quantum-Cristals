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

#----parametres du probleme---- 
L=2
N=3
nd = 11
K=np.linspace(-pi/L,pi/L,nd)
X0=[-4 for k in K]

#----la matrice B----
def B():
    B1=np.zeros( (2*N+1,2*N+1),dtype=complex )
    for n in range(2*N+1):
        B1[n,2*N-n]=1
    return(B1)

#----la matrice Ak----
def A(k):
    A1=np.zeros( (2*N+1,2*N+1),dtype=complex )
    for n in range(2*N+1):
        for m in range(2*N+1):
            if n+m==2*N :
                A1[n,m] = -4*j*(n+m-2*N)*sin(((1+2*(n+m-2*N))/2)*pi)/(pi*(4*(n+m-2*N)**2-1))
            else :
                A1[n,m] = (4*pi/L)*(m-N)*(-(pi/L)*(n-N)+k)+k**2-4*j*(n+m-2*N)*sin(((1+2*(n+m-2*N))/2)*pi)/(pi*(4*(n+m-2*N)**2-1))
    return(A1)

#----la matrice Sk----
def S(k):
    return( np.dot(nl.inv(B()),A(k)) )

#----les valeurs propres de Sk----
def VP(k):
    D, U = nl.eig(S(k))
    D.sort()
    return (D)

#----resolution et affichage----
for n in range(2*N+1):
    Yn=[re(VP(k)[n]) for k in K]
    plt.plot(K,Yn)
    plt.plot(X0,Yn)

plt.xlabel('k')
plt.ylabel('E_k,n')

plt.show()
#--------------------
#----2eme methode----
#--------------------
def e(n):
    return (lambda x : (1/sqrt(L))*exp(j*2*n*pi*x/L) )
def de(n):
    return (lambda x : (j*2*n*pi/(L*sqrt(L)))*exp(j*2*n*pi*x/L) )
Vper = lambda x : sin(2*pi*x/L)

def a(k,m,n): #calcule a_k(em,en)
    T1 = integrate.quad (RE(times(de(m),de(n))) , -L/2 , L/2 )[0] + j*integrate.quad(IM(times(de(m),de(n))) , -L/2 , L/2 )[0]
    T2 = - 2*j*k*(integrate.quad( RE(times(de(m),e(n))) , -L/2 , L/2 )[0] + j*integrate.quad( IM(times(de(m),e(n))) , -L/2 , L/2 )[0])
    T3 = (k**2)*(integrate.quad( RE(times(e(m),e(n))), -L/2 , L/2 )[0]+j*integrate.quad( IM(times(e(m),e(n))), -L/2 , L/2 )[0])
    T4 = integrate.quad(RE(times(times(Vper,e(m)),e(n))), -L/2 , L/2 )[0] +j*integrate.quad(IM(times(times(Vper,e(m)),e(n))), -L/2 , L/2 )[0]
    T = T1 + T2 + T3 + T4
    return(T)

def b(m,n):
    B = integrate.quad ( RE(times(e(m),e(n))), -L/2 , L/2 )[0] +j*integrate.quad ( IM(times(e(m),e(n))), -L/2 , L/2 )[0]
    return(B)

def B_2():
    B1=np.zeros( (2*N+1,2*N+1),dtype=complex )
    for m in range(2*N+1):
        for n in range(2*N+1):
            B1[m,n]=b(n-N,m-N)
    return(B1)

def A_2(k):
    A1=np.zeros( (2*N+1,2*N+1),dtype=complex )
    for m in range(2*N+1):
        for n in range(2*N+1):
            A1[m,n]=a(k,n-N,m-N)
    return(A1)

def S_2(k):
    return( np.dot(nl.inv(B_2()),A_2(k)) )

def VP_2(k):
    D, U = nl.eig(S_2(k))
    D.sort()
    return (D)

Y=np.zeros((2*N+1,len(K)))
for i in range(nd) :
    V=VP_2(K[i])
    for n in range(2*N+1):
        Y[n,i]=re(V[n])

for n in range(2*N+1):
    plt.plot(K,Y[n,:])
    plt.plot(X0,Y[n,:])

plt.xlabel('k')
plt.ylabel('E_k,n')

plt.show()


    

    


            
