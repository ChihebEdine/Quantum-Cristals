# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:00:15 2018

@author: Régis
"""

import math as m
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import cmath as cm

L=10; # Taille du domaine étudié
a1=10/5; # Période de répartition des atomes
C=L*1./a1;
a2=10/4; # perdiode de la deuxième distribution
N=100; # Nombre de fonctions de la base de Galerkine
epsi_F=2; # Niveau de Fermi
m1=20; # Poids de la réparition 1 des atomes
m2=0; # Poids de la réparition 2 des atomes
c=5;

m_l = 4;


erreur=10**(-6) # Critère d'arrêt de l'algorithme : ||V_{per,n}-V_{per,n-1}||_{L^2(R)}<erreur

#== Définition des vecteurs de base

def exp_j(x,j):
    return cm.exp(2*cm.pi*complex(0,1)*x*j/L)/cm.sqrt(L)
    
#==
   
#== Algorithme de FFT
    
def FFT(f,X,N):#N doit etre pair
    
    Xdiscret=np.arange(X[0],X[1],L*1./N)
    a=np.zeros(N)
    for l in range(N):
        a[l]=f(Xdiscret[l])
    A=np.fft.ifftshift(a)
    B=(1./N)*np.fft.fft(A)
    C=np.fft.fftshift(B)*cm.sqrt(L)
    for l in range(N):
        if abs(C[l])<=10**(-5):
            C[l]=0
    return C
    
def IFFT(M):
    N=len(M)
    M=np.fft.ifftshift(M)
    B=np.fft.ifft(M)
    B=N*np.fft.fftshift(B)/np.sqrt(L)
    return B.real

#==

#== Définition du potentiel de départ

def Vper(x):
    return 0
    

def norme(C1,C2):
    S=0
    for i in range(len(C1)):
        S+=(C1[i]-C2[i])**2
    return np.sqrt(S)
    
def norme2(X):
    n=len(X)
    s=0
    for i in range(n):
        s+=abs(X[i])**2
    return (np.sqrt(s))

# coef_Vper : liste de coefficients de Vper, fréquences de -N à N

def remplissage_matrice(coef_Vper):
    M=np.zeros((2*N+1,2*N+1), dtype=complex)
    for i in range(2*N+1):
        for j in range(2*N+1):
            M[i,j]+=coef_Vper[i-j+2*N]/np.sqrt(L)
            if i==j :
                M[i,j]+=(2*np.pi*(i-N)/L)**2
    D, U = nl.eig(M);
    sort_index = np.argsort(D)
    D.sort()
    FP=[]
    for i in range(len(D)):
        VPi = U[:,sort_index[i]];
        FP.append(VPi)
    return FP,D
    
    
# coef_Vper : liste de coefficients de Vper, fréquences de -N à N
# FP : Liste de liste de coefficients des fonctions propres de la matrice de masse, fréquences de -N à N
# D : Liste de valeurs propres correspondantes aux vecteurs propres    
      

def nouveau_coef_Vper(FP,D):
    COEF=[]
    DIRAC1=[]
    DIRAC2=[]
    RHO=[]
    for k in range(4*N+1):
        S=0
        if True :        
            for n in range(2*N+1):
                if D[n]<=epsi_F : # Critère sur le niveau d'énergie
                    for a in range(2*N+1):
                        if (a-k+2*N>=0 and a-k+2*N<2*N+1):
                            S+=FP[n][a-k+2*N].conjugate()*FP[n][a]/(norme2(FP[n][:])**2)
            S/=np.sqrt(L)
            RHO.append(S)
            M1=0
            M2=0
            for i in range(int(L*1./a1)):
                M1+=cm.exp(1j*np.pi*(k-2*N)*i*a1*1./L)
            for i in range(1,int(L*1./a2)):
                M2+=cm.exp(-1j*np.pi*(k-2*N)*i*a2*1./L)
            M1*=m1*1./L
            M2*=m2*1./L
            DIRAC1.append(M1)
            DIRAC2.append(M2)
            S-=M1
            S-=M2
            S/=(2*np.pi*np.pi*(k-2*N)*(k-2*N)/(L*L)+m_l*m_l)
        COEF.append(S)
    return COEF,DIRAC1,DIRAC2,RHO
    
coef_Vper1=[0]*(4*N+1)
compteur=0
RHO=[0]*(4*N+1)

while(True):
    FP,D=remplissage_matrice(coef_Vper1)
    coef_Vper2,DIRAC1,DIRAC2,RHO1=nouveau_coef_Vper(FP,D)
    print compteur
    compteur+=1
    if norme(RHO,RHO1)<=erreur or compteur>=100:
        break
    coef_Vper1=coef_Vper2
    RHO=RHO1
    

X=np.linspace(-L/2,L/2,4*N+1)
Y=IFFT(coef_Vper2)


DIRAC1=IFFT(DIRAC1)
DIRAC2=IFFT(DIRAC2)
RHO1=IFFT(RHO1) # Méthode par IFFT pour la densité électronique

for i in range(20):
    print D[i].real
    
liste_VP=np.zeros((2*N+1,2*N+1),dtype=list)
for i in range(2*N+1):
    if D[i]<=epsi_F:
        liste_VP[i,:]=IFFT(FP[i])
        
rho_elec=[]
for i in range(2*N+1):
    S=0
    for j in range(2*N+1):
        S+=liste_VP[j][i]*liste_VP[j][i].conjugate()
    rho_elec.append(S)


plt.subplot(5,1,1)
plt.title('Potentiel')
plt.xlabel('x')
plt.ylabel('Vper(x)')
plt.plot(X,Y)


plt.subplot(5,1,2)
plt.title('Densite atomique 1')
plt.xlabel('x')
plt.ylabel('Distribution 1')
plt.plot(X,DIRAC1)
plt.show()

plt.subplot(5,1,3)
plt.title('Densite atomique 2')
plt.xlabel('x')
plt.ylabel('Distribution 2')
plt.plot(X,DIRAC2)
plt.show()

plt.subplot(5,1,4)
plt.title('Densite atomique totale')
plt.xlabel('x')
plt.ylabel('rho_nucc(x)')
plt.plot(X,DIRAC1+DIRAC2)
plt.show()

#X=np.linspace(-L/2,L/2,2*N+1) # Pour voir rho_elec à la place de RHO1

plt.subplot(5,1,5)
plt.title('Densite electronique')
plt.xlabel('x')
plt.ylabel('rho_elec(x)')
plt.plot(X,RHO1)
plt.show()














