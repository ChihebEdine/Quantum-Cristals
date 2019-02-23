import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nl
from math import*
from random import*

#parametres -----------------------------------------------------
L=50                         # largeur du domaine 
N=50                 # parametre de discrétisation
m=5                      # parametre de l'equation du Poisson
Ef=1                             # niveau de Fermi
C=1                              # rho atom
V0=np.zeros(4*N+1,dtype=complex) # potentiel initial
epsilon=0.03                    # condition de convergence
nd=3                            # nombre de défaut

#random - perturbation --------------------------------------------------------
#centre=[ uniform(-L/2,L/2) for i in range(nd) ]
#amplitude = [ uniform(0,0.02) for i in range(nd) ]
centre=[0]
amplitude=[0.01]
# la norme 2 d'un vecteur complexe ---------------------------------------
def norme2(X):
    n=len(X)
    s=0
    for i in range(n):
        s+=abs(X[i])**2
    return (sqrt(s))

# matrice A(V) ----------------------------------------------------

def A(V):
    A=np.zeros((2*N+1,2*N+1),dtype=complex)
    for a in range(2*N+1):
        for b in range(2*N+1):
            A[a,b]=V[a-b+2*N]
            if (a==b) :
                A[a,b]+=2*(((a-N)*pi)/L)**2
    return(A)
# roh-elec -----------------------------------------------------
def roh_elec(E,P):
    r=np.zeros(4*N+1,dtype=complex)
    for k in range(4*N+1):
        s=0
        for n in range(2*N+1):
            if(E[n]<Ef):
                sk=0
                for a in range(2*N+1):
                    if (a-k+2*N >= 0 and a-k+2*N < 2*N+1):
                        sk+=(P[n,a]*P[n,a-k+2*N].conjugate())/(norme2(P[n,:])**2)
                s+=sk
        s= (1/(sqrt(L))) * s
        r[k]=s
    return(r)
    
                
# le vecteur des coefficents de fourier du potentiel ----------- 
def V(E,P,centre,amplitude):
    V=np.zeros(4*N+1,dtype=complex)    
    for k in range(4*N+1):
        s=0
        for n in range(2*N+1):
            if(E[n]<Ef):
                sk=0
                for a in range(2*N+1):
                    if (a-k+2*N >= 0 and a-k+2*N < 2*N+1):
                        sk+=(P[n,a]*P[n,a-k+2*N].conjugate())/(norme2(P[n,:])**2)
                s+=sk
        s= (1/(sqrt(L))) * s
        if ( k == 2*N ):
            ss=0
            for i in range(len(centre)):
                ss+=amplitude[i]
            s-=sqrt(L)*(C+ss)
        else :
            ss=0
            for i in range(len(centre)):
                ss+=sin(amplitude[i]*pi*(k-2*N))*(cos( ((2*pi)/L) * centre[i] * (k-2*N) )-1j*sin(((2*pi)/L) * centre[i] * (k-2*N))  ) 
            s-=sqrt(L)/((k-2*N)*pi)*ss
        if(m==0 and k==2*N):
            V[k]=0
        else:
            V[k]=2*s/(m**2+(2*pi*(k-2*N)/L)**2)
            
    return(V)


# resolution ------------------------------------------------------------
E, P = nl.eig(A(V0))
V2=V(E,P,centre,amplitude)
while (True):
    V1=V2
    E,P = nl.eig( A(V2) )
    V2=V(E,P,centre,amplitude)
    d=norme2(V2-V1)
    print(d)
    if(d<epsilon):
        break
    
#affichage --------------------------------------------------------------
R=roh_elec(E,P)

Vc=(4*N+1)*V2
R=(4*N+1)*R

Vc=Vc/sqrt(L)
R=R/sqrt(L)

Vc=np.fft.ifftshift(Vc)
R=np.fft.ifftshift(R)

Vf=np.fft.ifft(Vc)
Rf=np.fft.ifft(R)

Vf=np.fft.fftshift(Vf)
Rf=np.fft.fftshift(Rf)



X=np.array([  -L/2+k*L/len(Vf) for k in range(len(Vf)) ])
Y=np.array([  Vf[k].real for k in range(len(Vf))  ])
Z=np.array([  Rf[k].real for k in range(len(Rf))  ])

plt.subplot(2,1,1)
plt.plot(X,Y)
plt.ylabel('le potentiel')

plt.subplot(2,1,2)
plt.plot(X,Z)
plt.xlabel('x')
plt.ylabel('La densité des éléctrons ')
print('amplitude = ', amplitude )
print('centre = ',centre )
plt.show()


