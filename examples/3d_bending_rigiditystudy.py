##############################################################################
# Authors: Andreu Fernandez Gallen and Joan Muñoz Biosca, contact email: fdzgallen@gmail.com
#
# This code implements the model presented in the article "Vorticity-Stream vector formulation 
# for 3-dimensional viscous flow of deformable bodies.
#
# This model simulates a biological membrane characterized in a Poiseuille flow by its bending 
# energy and area+volume conservation inside a fluid flow in 3D.
# The fluid flow is computed by solving 2 Poisson equations to obtain the 
# stream function and the voriticity, which we use to compute the flow.
##############################################################################
import numpy as np 
import os.path
import sys
from copy import deepcopy
import time
import os
from pathlib import Path
import pickle 
import matplotlib
matplotlib.use('Agg')

from src.plots_and_outputs import save_vti_file, dump,output_folders

from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib._color_data as mcd
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

from scipy.signal import butter, lfilter, freqz
from scipy import signal
from scipy.stats import norm
from scipy import ndimage

from shutil import copyfile
from datetime import datetime


#----------------------------------------  Constants and variables fo the simulation  ----------------------------------------


simulation="output/descentered_30" #Simulation = Output directory of data
print(simulation)


M= 1.0       # mobility
dt= 0.0001       # temporal resolution
h= 1.0           # spatial resolution
tdump= 20000     # phi dump period
Tend= tdump*240   # number of steps
t0= tdump*10     # time to relax the interface
t1 = tdump*20   # time given to form the interface after turning on the Area/volume conservation
Nx= 55          # lattice size X
Ny= 55           # lattice size Y
Nz= 65           # lattice size Z
unity=np.ones((Nx,Ny,Nz),float)
Nx0= int((Nx-1)/2)     # lattice size X
Ny0= int((Ny-1)/2-3)       # lattice size Y
Nz0= int((Nz-1)/2)       # lattice size Z 
R = int(((Ny-1)/2))    #Radious of the cylinder channel
kappa= 1.0              #Bending rigidity
viscositycontrast = 1.0 #If =! 1, the viscosity definition has to be switch on
nit = 5     #iterative steps taken for the Possion solver, low is good as the differences between time-steps are low
eps= 1.0         # interfacial width of the phase field
C0= 0.0          # spontanous curvature 
alpha2= 0.5 # Strength of the lagrange mutiplier effect for AREA
beta= -0.01    # Strength of the lagrange mutiplier effect for volume 
viscosityliq = 1.0 #viscosity of the sourrounding liquid
Pspeed= 0.90 #Poiseuille speed at the center of the channel
deltaPdl= (Pspeed*8.0*viscosityliq)/(R**2)  #difference of pressure given the parameters
cte= deltaPdl/(viscosityliq) #constant later used constantly, we define it to avoid repetitive divisions which are slower than multiplications
anglebeta=-((np.pi)/2.0)*1.3  #angle to define starting cell that is rotated
anglegamma=-((np.pi)/2.0)*1.0 #angle to define starting cell that is rotated
 
 


#----------------------------------------  Functions  ----------------------------------------

def lap_roll(u,mask):  #laplatian using roll of a vector u. Returns a vector
    lapu = np.zeros_like(u)
    lapu[mask] = (1/(h*h))*(np.roll(u,1,0)[mask]+np.roll(u,-1,0)[mask]+np.roll(u,1,1)[mask]+np.roll(u,-1,1)[mask]+np.roll(u,1,2)[mask]+np.roll(u,-1,2)[mask]-6*u[mask])
    return lapu

def grad_roll(u,mask):  #gradient using roll of a given field u. Returns a vector of 3 vectors [0][1][2]
    gradux = np.zeros_like(u)
    graduy = np.zeros_like(u)
    graduz = np.zeros_like(u)
    gradux[mask] = (0.5/h)*(-np.roll(u,1,0)[mask]+np.roll(u,-1,0)[mask])
    graduy[mask] = (0.5/h)*(-np.roll(u,1,1)[mask]+np.roll(u,-1,1)[mask])
    graduz[mask] = (0.5/h)*(-np.roll(u,1,2)[mask]+np.roll(u,-1,2)[mask])
    gu = [gradux,graduy,graduz]
    return gu
 
#Solving Poisson equation with given function b of vector p, using cylindrical geometry. Returns the solution vector of the equation
def poisson(poiss,b,mask): 
    pn = np.empty_like(poiss)
    for q in range(nit):
        pn = poiss.copy()
        poiss[mask] = (1.0/6.0)*(np.roll(pn,1,0)[mask] + np.roll(pn,-1,0)[mask] + np.roll(pn,1,1)[mask] + np.roll(pn,-1,1)[mask] + np.roll(pn,1,2)[mask] + np.roll(pn,-1,2)[mask] - h*h*b[mask])  
    return poiss


#Computing psi function. Returns vector psi, and laplatian of vector psi (vector)
def compute_psi(phi0,lap_phi,mask,unity,eps,C0):
    psi_sc = np.zeros_like(phi0)
    psi_sc[mask]=phi0[mask]*(-unity[mask]+phi0[mask]*phi0[mask])-eps*eps*lap_phi[mask]-eps*C0*(unity[mask]-phi0[mask]*phi0[mask])
    lap_psi= lap_roll(psi_sc,mask)
    return psi_sc,lap_psi

#Computing mu. Returns vector psi, and laplatian of vector psi (vector)
def compute_mu_b(phi0,psi_sc,lap_psi,mask):
    mu_b = np.zeros_like(phi0)
    mu_b[mask]=(3.0*(2.0**0.5)/(4.0*eps**3))*kappa*(psi_sc[mask]*(-unity[mask]+3.0*phi0[mask]*phi0[mask]-2*eps*C0*phi0[mask])-eps*eps*lap_psi[mask])
    lap_mu_b = lap_roll(mu_b,mask)
    return mu_b, lap_mu_b

def compute_mu_multipl(phi, lap_phi, sigma1, sigma2, sigmav,mask):
    mu_multipl = np.zeros_like(phi)
    aux1 = grad_roll(phi,mask)
    mu_multipl[mask]= (sigma2*3*(2.0**0.5)/(4.0))*(((phi[mask]*(phi[mask]**2-unity[mask]))/eps) -eps*lap_phi[mask])
    mu_multipl[mask] = mu_multipl[mask] + sigma2*((2**0.5)*3*eps/4)*(np.sqrt(aux1[0][mask]*aux1[0][mask]+aux1[1][mask]*aux1[1][mask]+aux1[2][mask]*aux1[2][mask]))
    lap_mu_multipl = lap_roll(mu_multipl,mask)
    return mu_multipl, lap_mu_multipl

# Computing Area. Integrating the gradient of phi is equivalent to compute the area of the membrane because
# the bulk in the system has gradient 0 therefore only the membrane contributes
def get_area(phi,mask):
    aux1 = grad_roll(phi,mask)
    return ((2.0**0.5)*3.0*eps/8.0)*(np.sum(aux1[0][mask]*aux1[0][mask]+aux1[1][mask]*aux1[1][mask]+aux1[2][mask]*aux1[2][mask]))

#Computing Area. phi > 0 is a boolean array, so `sum` counts the number of True elements
def get_area3(phi,mask): 
    return ((2.0**0.5)*3.0/(eps*16.0))*(((phi[mask]**2-unity[mask])**2).sum()) #1 is a matrix with 1 - multiplying two matrices is not dot product! is just multiply number per number

#Computing Volume. phi > 0 is a boolean array, so `sum` counts the number of True elements
def get_volume(phi,mask):
    return ((((booll2>0.0).sum() + (phi[mask]).sum())/2))
 
#Lagrange Multiplier Area
def compute_multiplier_2(A0,Area): 
    return  alpha2 * (Area - A0)
 
#Lagrange Multiplier Volume
def compute_multiplier_volume(V0,Volume):
    return  beta * (Volume - V0) 

#Lagrange Multiplier for create interphase
def compute_multiplier_global2_withoutflow(gradphi,grad_lapmu,grad_laparea,mask):
    sigma_N2 = M*np.sum(gradphi[0][mask]*grad_lapmu[0][mask]+gradphi[1][mask]*grad_lapmu[1][mask]+gradphi[2][mask]*grad_lapmu[2][mask])
    sigma_D = ((2.0**0.5)*3.0*eps/4.0)*M*np.sum(gradphi[0][mask]*grad_laparea[0][mask]+gradphi[1][mask]*grad_laparea[1][mask]+gradphi[2][mask]*grad_laparea[2][mask])
    sigma = (sigma_N2)/sigma_D 
    return sigma

#Lagrange Multiplier during temporal evolution of phi
def compute_multiplier_global2(gradphi,grad_lapmu,grad_laplapphi,v_x,v_y,v_z,mask):
    gradiii = grad_roll(v_x*gradphi[0]+v_y*gradphi[1]+v_z*gradphi[2],mask)
    sigma_N1 = np.sum(-gradphi[0][mask]*gradiii[0][mask]-gradphi[1][mask]*gradiii[1][mask]-gradphi[2][mask]*gradiii[2][mask])
    sigma_N2 = M*np.sum(gradphi[0][mask]*grad_lapmu[0][mask]+gradphi[1][mask]*grad_lapmu[1][mask]+gradphi[2][mask]*grad_lapmu[2][mask])
    sigma_D = ((2.0**0.5)*3.0*eps/4.0)*M*np.sum(gradphi[0][mask]*grad_laplapphi[0][mask]+gradphi[1][mask]*grad_laplapphi[1][mask]+gradphi[2][mask]*grad_laplapphi[2][mask])
    sigma = (sigma_N1+sigma_N2)/sigma_D 
    return sigma

#Computing viscosity
def compute_visco(phi,mask):
    visco = np.zeros_like(phi)
    visco=viscosityliq*0.5*(unity-phi)+viscosityliq*viscositycontrast*0.5*(unity+phi) #viscosityliq*viscositycontrast = viscosity cell
    return visco

def do_rotation_y_x(i,k):
    return ((i-Nx0)*np.cos(anglebeta) -(k-Nz0)*np.sin(anglebeta))

def do_rotation_y_z(i,k):
    return ((i-Nx0)*np.sin(anglebeta) +(k-Nz0)*np.cos(anglebeta))

#There is no rotation of y component if the rotation is done around the y-axis
 
def do_rotation_yz_x(i,j,k):
    return ((i-Nx0)*np.cos(anglebeta)*np.cos(anglegamma) +(j-Ny0)*np.sin(anglegamma)*np.cos(anglebeta) - (k-Nz0)*np.sin(anglebeta))
    
def do_rotation_yz_y(i,j,k):
    return (-(i-Nx0)*np.sin(anglegamma) +(j-Ny0)*np.cos(anglegamma))

def do_rotation_yz_z(i,j,k):
    return ((i-Nx0)*np.cos(anglegamma)*np.sin(anglebeta) +(j-Ny0)*np.sin(anglegamma)*np.sin(anglebeta) + (k-Nz0)*np.cos(anglebeta))

    
#----------------------------------------  Creating folders  ----------------------------------------

print(os.getcwd())

output_folders(simulation)

#Lets create a copy of the code in the ooutput to be able to double check in the future the version used for those results
dodo=os.path.join("./"+simulation+"/"+os.path.basename(__file__))
copyfile(__file__,dodo)  


#----------------------------------------  Main  ----------------------------------------

#Variables and lists to store computed observables etc
phi=np.zeros((Nx,Ny,Nz),float)
phi = -unity
w_x=np.zeros((Nx,Ny,Nz),float)
w_y=np.zeros((Nx,Ny,Nz),float)
w_z=np.zeros((Nx,Ny,Nz),float)
w_mod = np.zeros((Nx,Ny,Nz),float)
v_x=np.zeros((Nx,Ny,Nz),float)
v_y=np.zeros((Nx,Ny,Nz),float)
v_z=np.zeros((Nx,Ny,Nz),float)
v_mod = np.zeros((Nx,Ny,Nz),float)
A_x=np.zeros((Nx,Ny,Nz),float)
A_y=np.zeros((Nx,Ny,Nz),float)
A_z=np.zeros((Nx,Ny,Nz),float)
A_mod = np.zeros((Nx,Ny,Nz),float)
w_x0 = np.zeros((Nx,Ny,Nz),float)
w_y0 = np.zeros((Nx,Ny,Nz),float)
w_z0 = np.zeros((Nx,Ny,Nz),float)  
A_x0 = np.zeros((Nx,Ny,Nz),float) 
A_y0 = np.zeros((Nx,Ny,Nz),float)
A_z0 = np.zeros((Nx,Ny,Nz),float) 
lap_mu_multipl = np.zeros((Nx,Ny,Nz),float)
xi = []
omega = []
xi_y = []
omega_y = []
xi_x = []
omega_x = []
xi_z = []
omega_z = []
booll2 = np.zeros((Nx,Ny,Nz))
booll = np.zeros((Nx,Ny))
newbooll2 = np.zeros((Nx,Ny,Nz))
newbooll = np.zeros((Nx,Ny))
pos = []
sigmavmatrix = np.zeros_like(phi)

# Two Radii of the disc
R1=14.0
R2=0.29*R1


#Creating boolean matrix booll. It indicates the whole cylindrical system
for x in reversed (range (Nx)):
    for y in range (0,Ny):
        for k in range (0,Nz):
            if((x-R)**2 + (y-R)**2 <= R*R):
                aux = np.array([x,y])
                pos.append(aux)
                booll2 [x,y,k] = 1
for x in reversed (range (Nx)):
    for y in range (0,Ny):
        if((x-R)**2 + (y-R)**2 <= R*R):
            aux = np.array([x,y])
            pos.append(aux)
            booll [x,y] = 1

#Creating boolean matrix newbooll. It indicates the points where you calculate evolutions. Newbooll - booll = boundary points of the cylinder
newbooll2 = booll2.copy()
newbooll2[int(Nx-1),int((Nx-1)/2),:] = 0; newbooll2[int((Nx-1)/2),int(Nx-1),:] = 0; newbooll2[0,int((Nx-1)/2),:] = 0; newbooll2[int((Nx-1)/2),0,:] = 0
 
 

#Creating boolean matrix newbooll. It indicates the points where you calculate evolutions. Newbooll - booll = boundary points of the cylinder
newbooll = booll.copy()
newbooll[int(Nx-1),int((Nx-1)/2)] = 0; newbooll[int((Nx-1)/2),int(Nx-1)] = 0; newbooll[0,int((Nx-1)/2)] = 0; newbooll[int((Nx-1)/2),0] = 0
 
for i in range(1,Nx-1):
    for j in range(1, Ny-1):
        if(booll[i+1][j] == 0 or booll[i-1][j] == 0 or booll[i][j+1] == 0 or booll[i][j-1] == 0):
            newbooll[i][j] = 0
            newbooll2[i,j,:] = 0


#--------------------  Initial conditions  --------------------

for i in range(0,Nx):
    for j in range(0,Ny):
        if(booll[i][j]): #just if you ara inside the cylinder
            for k in range(0,Nz):
            # elipsoidal initial conditions
                if ((float(i)-Nx0)*(float(i)-Nx0)/(R1**2)+(float(j)-Ny0)*(float(j)-Ny0)/(R1**2)+(float(k)-Nz0)*(float(k)-Nz0)/(R2**2)>1.0):
            # sphereinitial conditions
                #if ((float(i)-Nx0)*(float(i)-Nx0)+(float(j)-Ny0)*(float(j)-Ny0)+(float(k)-Nz0)*(float(k)-Nz0)>R1*R1): 
            # rotated elipsoidal initial conditions
                #rot_i = do_rotation_yz_x(i,j,k)
                #rot_j = do_rotation_yz_y(i,j,k)
                #rot_k = do_rotation_yz_z(i,j,k)
                #if ((rot_i)*(rot_i)/(R1**2)+(rot_j)*(rot_j)/(R1**2)+(rot_k)*(rot_k)/(R2**2)>1.0): 
                    phi[i,j,k]= -1.0
                else:
                    phi[i,j,k]= 1.0

                #conditions for velocity vorticity vector potential 
                w_x[i,j,k] = (-cte/2.0)*(float(j)-R)
                w_y[i,j,k] = (cte/2.0)*(float(i)-R)
                A_x[i,j,k] = (cte/4.0)*((float(j)**3)/3.0 - R*(float(j)**2) + (float(j)*(R**2))/2.0)
                A_y[i,j,k] = (-cte/4.0)*((float(i)**3)/3.0 - R*(float(i)**2) + (float(i)*(R**2))/2.0)
                v_z[i,j,k] = (cte/4.0)*(R**2 - (float(i)-R)**2 - (float(j)-R)**2)   
                        


#--------  First temporal loop - Interphase formation  ---------

A_x0 = A_x.copy()
A_y0 = A_y.copy()
w_x0 = w_x.copy()
w_y0 = w_y.copy() 
energyb=[]
energym=[]
areatalt=[]
vol=[]
areat=[]
areat2=[]
areatot=[]
volred = []
sav = []
Sigma=0.0 #Lagrange multiplier for the area starts at 0 while the diffuse interfase is generated
Sigma2=0.0
Sigma3=0.0 
maskT=newbooll2>0.5
maskvol = booll2>0.5


#print("Initial areaasaco -> ", areainiasaco)

start = time.perf_counter()

Area=get_area(phi,maskT)
Area2=get_area3(phi,maskT) 
Areatot= Area+Area2
volume = get_volume(phi,maskvol)
area2 = 0.0
area3 = 0.0
volini=(4.0/3.0)*np.pi*(R1**3)

print("Initial compuarea -> ", Areatot) 
print("Initial compuvol -> ", volume)
print("Initial volini -> ", volini)


for t in range(0,t1):
    lap_phi = lap_roll(phi,maskT)
    lap_lap_phi=lap_roll(lap_phi,maskT)
    psi_sc,lap_psi =compute_psi(phi,lap_phi,maskT,unity,eps,C0)
    mu_b,lap_mu_b = compute_mu_b(phi,psi_sc,lap_psi,maskT)

    if(t < t0):
        phi[maskT] += dt*M*(lap_mu_b[maskT])
        A01= get_area(phi,maskT)
        A02= get_area3(phi,maskT)
        V0=get_volume(phi,maskvol) 
    else:
        v=get_volume(phi,maskvol)
        area1=get_area(phi,maskT)
        area3=get_area3(phi,maskT)
        grad_lapmu_b = grad_roll(lap_mu_b,maskT) 
        Sigma2 = compute_multiplier_2(A01+A02,area1+area3) 
        Sigma3 = compute_multiplier_volume(V0,v)
        sigmavmatrix = Sigma3*unity
        mu_multipl,lap_mu_multipl = compute_mu_multipl(phi, lap_phi, Sigma, Sigma2, Sigma3, maskT)
        phi[maskT] += dt*M*(lap_mu_b[maskT]+lap_mu_multipl[maskT]+sigmavmatrix[maskT])

    #Print data- Only phi changes: velocity vorticity and A remain constant
    if (t % tdump == 0): 
        v_mod = (v_x*v_x + v_y*v_y + v_z*v_z)**(0.5)
        w_mod = (w_x*w_x + w_y*w_y + w_z*w_z)**(0.5)
        A_mod = (A_x*A_x + A_y*A_y + A_z*A_z)**(0.5)

        dump(t,start,Tend, tdump, simulation, energyb, energym, h, eps,unity, booll2, kappa,
        maskT,maskvol,phi,xi, omega, xi_x, xi_y,xi_z,omega_x,omega_y,omega_z,
        vol, areat, areat2, areatot, areatalt, volred, sav,psi_sc,Sigma2,
        Nx,Ny,Nz,Nx0,Ny0,Nz0, w_mod, A_mod, v_x,v_y,v_z, v_mod,
        A_x,A_y,A_z,A_x0,A_y0,A_z0,w_x,w_y,w_z,w_x0,w_y0,w_z0)
        save_vti_file(t,phi,v_x,v_y,v_z, Nx, Ny, Nz,booll2,newbooll2,simulation)
        start = time.perf_counter()        
        

#--------------------  Main temporal loop  --------------------
 
for t in range(t1,Tend):

    #0-Calculating parameters
    maskphi = phi>0.0
    lap_phi = lap_roll(phi,maskT)
    psi_sc,lap_psi = compute_psi(phi,lap_phi,maskT,unity,eps,C0)
    #viscosity = compute_visco(phi,maskT) #just if the contrast is =! 1
    area1=get_area(phi,maskT)
    area3=get_area3(phi,maskT)
    v=get_volume(phi,maskvol)  
    Sigma2 = compute_multiplier_2(A01+A02,area1+area3)
    Sigma3 = compute_multiplier_volume(V0,v)
    sigmavmatrix = Sigma3*unity
    gradphi = grad_roll(phi,maskT)
    mu_b,lap_mu_b = compute_mu_b(phi,psi_sc,lap_psi,maskT)
    grad_lapmu_b = grad_roll(lap_mu_b,maskT)
    mu_mutipl,lap_mu_multipl = compute_mu_multipl(phi,lap_phi, Sigma, Sigma2, Sigma3,maskT)
    gradmu = grad_roll(mu_b + mu_mutipl,maskT)

    #1-Calculating w with poisson
    w_x = poisson(w_x, (gradmu[2]*gradphi[1] - gradmu[1]*gradphi[2])/viscosityliq,maskT) #if contrast is =! 1 replace for viscosity
    w_y = poisson(w_y, (gradmu[0]*gradphi[2] - gradmu[2]*gradphi[0])/viscosityliq,maskT)
    w_z = poisson(w_z, (gradmu[1]*gradphi[0] - gradmu[0]*gradphi[1])/viscosityliq,maskT)

    #2-Calculating A with poisson
    A_x = poisson(A_x, -w_x,maskT)
    A_y = poisson(A_y, -w_y,maskT)
    A_z = poisson(A_z, -w_z,maskT)

    #3-Calculating grad of A
    grad_Ax = grad_roll(A_x,maskT)
    grad_Ay = grad_roll(A_y,maskT)
    grad_Az = grad_roll(A_z,maskT)

    #4-Calculating velocity
    v_x = grad_Az[1] - grad_Ay[2]
    v_y = grad_Ax[2] - grad_Az[0]
    v_z = grad_Ay[0] - grad_Ax[1]  

    #5-Calculating temporal evolution of phi
    
    phi[maskT] += dt*M*(lap_mu_b[maskT]+lap_mu_multipl[maskT]+sigmavmatrix[maskT])
    phi[maskT] += -dt*(v_x[maskT]*gradphi[0][maskT] +v_y[maskT]*gradphi[1][maskT] +v_z[maskT]*gradphi[2][maskT]) #advection term 

    #Print data
    if (t % tdump == 0):
        v_mod = (v_x*v_x + v_y*v_y + v_z*v_z)**(0.5)
        w_mod = (w_x*w_x + w_y*w_y + w_z*w_z)**(0.5)
        A_mod = (A_x*A_x + A_y*A_y + A_z*A_z)**(0.5)  
        
        dump(t,start,Tend, tdump, simulation, energyb, energym, h, eps,unity, booll2, kappa,
        maskT,maskvol,phi,xi, omega, xi_x, xi_y,xi_z,omega_x,omega_y,omega_z,
        vol, areat, areat2, areatot, areatalt, volred, sav,psi_sc,Sigma2,
        Nx,Ny,Nz,Nx0,Ny0,Nz0,w_mod,A_mod, v_x,v_y,v_z, v_mod,
        A_x,A_y,A_z,A_x0,A_y0,A_z0,w_x,w_y,w_z,w_x0,w_y0,w_z0)

        save_vti_file(t,phi,v_x,v_y,v_z, Nx, Ny, Nz,booll2,newbooll2,simulation)
        start = time.perf_counter()


