import numpy as np 
import os.path
import sys
from copy import deepcopy
import time
import os
from pathlib import Path
import pickle
#import psutil
import matplotlib
matplotlib.use('Agg')

#from matplotlib.colors import DivergingNorm
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


#----------------------------------------  Constants  ----------------------------------------


M= 1.0       # mobility
dt= 0.0001       # temporal resolution
h= 1.0           # spatial resolution
Tend= 2000*2400   # number of steps
tdump= 20000     # phi dump period
t0= tdump*10     # time to relax the interface
t1 = tdump*20 
Nx= 55         # lattice size X
Ny= 55           # lattice size Y
Nz= 65           # lattice size Z
unity=np.ones((Nx,Ny,Nz),float)
Nx0= int((Nx-1)/2)     # lattice size X
Ny0= int((Ny-1)/2-3)       # lattice size Y
Nz0= int((Nz-1)/2)       # lattice size Z 
R= int(((Ny-1)/2)) 
kappa= 1.0 
viscositycontrast = 1.0 #If =! 1, the viscosity definition has to be switch on
nit = 5
eps= 1.0         # interfacial width
C0= 0.0          # spontanous curvature
alpha1= 0.5 # Strength of the lagrange mutiplier effect for AREA
alpha2= 0.5 # Strength of the lagrange mutiplier effect for AREA
beta= -0.01    # Strength of the lagrange mutiplier effect for volume
betabef = -0.001
nit=5
viscosityliq = 1.0
Pspeed= 0.90
deltaPdl= (Pspeed*8.0*viscosityliq)/(R**2)
cte= deltaPdl/(viscosityliq)
anglebeta=-((np.pi)/2.0)*1.3
anglegamma=-((np.pi)/2.0)*1.0

#p = psutil.Process()
#p.cpu_affinity([17,18])
#psutil.cpu_count()
#psutil.cpu_count(logical=False)
#len(psutil.Process().cpu_affinity()) 
simulation="descentered_30_4" #Simulation = Output directory of data
print(simulation)


#----------------------------------------  Functions  ----------------------------------------

def save_vti_file(phi,vx,vy,vz, nx, ny, nz):
    
    pc_lista_novo_phi = [] # criacao de uma lista com os valores do array 3D 'pc'
    pc_lista_novo_vx = []
    pc_lista_novo_vy = []
    pc_lista_novo_vz = []
    pc_lista_novo_cyl = []
    pc_lista_novo_syst = []
    
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                t_phi = phi[x,y,z]
                t_vx = vx[x,y,z]
                t_vy = vy[x,y,z]
                t_vz = vz[x,y,z]
                t_cyl = booll2[x,y,z]
                t_syst = newbooll2[x,y,z]
                pc_lista_novo_phi.append(t_phi)
                pc_lista_novo_vx.append(t_vx)
                pc_lista_novo_vy.append(t_vy)
                pc_lista_novo_vz.append(t_vz)
                pc_lista_novo_cyl.append(t_cyl)
                pc_lista_novo_syst.append(t_syst)
    
    pc_string_novo_phi = "    ".join([str(_) for _ in pc_lista_novo_phi]) # criacao de uma string com os valores da lista
    pc_string_novo_vx = "    ".join([str(_) for _ in pc_lista_novo_vx])
    pc_string_novo_vy = "    ".join([str(_) for _ in pc_lista_novo_vy])
    pc_string_novo_vz = "    ".join([str(_) for _ in pc_lista_novo_vz])
    pc_string_novo_cyl = "    ".join([str(_) for _ in pc_lista_novo_cyl])
    pc_string_novo_syst = "    ".join([str(_) for _ in pc_lista_novo_syst])
    file=os.path.join("./"+simulation+"/vti/tot_t="+str(t).zfill(8)+".vti")
    with open(file, "w" ) as my_file:
        my_file.write('<?xml version="1.0"?>')
        my_file.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        my_file.write('  <ImageData WholeExtent="0 '+str(nx)+' 0 '+str(ny)+' 0 '+str(nz)+'" Origin="0 0 0" Spacing ="1 1 1">\n')
        my_file.write('    <Piece Extent="0 '+str(nx)+' 0 '+str(ny)+' 0 '+str(nz)+'">\n') # dimensao da matriz x1 x2 y1 y2 z1 z2
        my_file.write('     <CellData>\n')
        my_file.write('     <DataArray Name="phi" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_phi)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="v_x" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_vx)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="syst" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_syst)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="v_y" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_vy)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="v_z" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_vz)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="cyl" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_cyl)
        my_file.write('\n         </DataArray>\n')
        my_file.write('      </CellData>\n')
        my_file.write('    </Piece>\n')
        my_file.write('</ImageData>\n')
        my_file.write('</VTKFile>\n')
        my_file.close() # fecha o ficheiro

#laplatian using roll of a vector u. Returns a vector
def lap_roll(u,mask): 
    lapu = np.zeros_like(u)
    lapu[mask] = (1/(h*h))*(np.roll(u,1,0)[mask]+np.roll(u,-1,0)[mask]+np.roll(u,1,1)[mask]+np.roll(u,-1,1)[mask]+np.roll(u,1,2)[mask]+np.roll(u,-1,2)[mask]-6*u[mask])
    return lapu


#gradient using roll of a given field u. Returns a vector of 3 vectors [0][1][2]
def grad_roll(u,mask):
    gradux = np.zeros_like(u)
    graduy = np.zeros_like(u)
    graduz = np.zeros_like(u)
    gradux[mask] = (0.5/h)*(-np.roll(u,1,0)[mask]+np.roll(u,-1,0)[mask])
    graduy[mask] = (0.5/h)*(-np.roll(u,1,1)[mask]+np.roll(u,-1,1)[mask])
    graduz[mask] = (0.5/h)*(-np.roll(u,1,2)[mask]+np.roll(u,-1,2)[mask])
    gu = [gradux,graduy,graduz]
    return gu

"""
#laplatian using cylinder geometry of a vector p. Returns a vector
def lap_roll(p,mask): 
    pn = np.zeros_like(p)
    #for q in range (3):
    pn = p.copy()
#    for i in range(1,Nx-1):
 #       for j in range(1,Ny-1):
  #          if(nb[i][j]):
   #             p[i,j,1:-1] = (1.0/h*h)*(pn[i+1,j,1:-1] + pn[i-1,j,1:-1] + pn[i,j+1,1:-1] + pn[i,j-1,1:-1] + pn[i,j,2:] + pn[i,j,0:-2] - 6.0*pn[i,j,1:-1]) #bulk
    #            p[i, j, 0] = (1.0/h*h)*(pn[i+1,j,0] + pn[i-1,j,0] + pn[i,j+1,0] + pn[i,j-1,0] + pn[i,j,1] + pn[i,j,-1] - 6.0*pn[i,j,0])
     #           p[i, j, -1] =  (1.0/h*h)*(pn[i+1,j,-1] + pn[i-1,j,-1] + pn[i,j+1,-1] + pn[i,j-1,-1] + pn[i,j,0] + pn[i,j,-2] - 6.0*pn[i,j,-1]) 

    return p

#Gradient of a vector u using cylindrical geometry. Returns a vector of 3 vectors [0][1][2]
def grad_roll(u,mask): 
    ux = np.zeros((Nx,Ny,Nz))
    uy = np.zeros((Nx,Ny,Nz))
    uz = np.zeros((Nx,Ny,Nz))
#    for i in range(1,Nx-1):
 #       for j in range(1,Ny-1):
 #           if(nb[i][j]):
  #              uz[i, j, 0] = (0.5*h)*(u[i,j,1] - u[i,j,-1])
   #             uz[i, j, -1] =  (0.5*h)*(u[i,j,0] - u[i,j,-2])
    #            for k in range(0,Nz): 
     #               ux[i,j,k] = (0.5*h)*(u[i+1,j,k] - u[i-1,j,k]) #bulk
      #              uy[i,j,k] = (0.5*h)*(u[i,j+1,k] - u[i,j-1,k])
       #             if(k != 0 and k != Nz-1):
        #                uz[i,j,k] = (0.5*h)*(u[i,j,k+1] - u[i,j,k-1])
    gu = [ux,uy,uz]
    return gu

#Solving Poisson equation with given function b of vector p, using cylindrical geometry. Returns the solution vector of the equation
def poisson(p,b,mask): 
    pn = np.zeros_like(p)
    #for q in range (3):
    pn = p.copy() 
   # for i in range(1,Nx-1):
       # for j in range(1,Ny-1):
            # if(nb[i][j]):
            # for k in range(1,Nz-1):
                # p[i,j,k] = (1.0/6.0)*(pn[i+1,j,k] + pn[i-1,j,k] + pn[i,j+1,k] + pn[i,j-1,k] + pn[i,j,k+1] + pn[i,j,k-1] - h*h*b[i,j,k]) #bulk
            # p[i, j, 0] = (1.0/6.0)*(pn[i+1,j,0] + pn[i-1,j,0] + pn[i,j+1,0] + pn[i,j-1,0] + pn[i,j,1] + pn[i,j,-1] - h*h*b[i,j,0])
            # p[i, j, -1] =  (1.0/6.0)*(pn[i+1,j,-1] + pn[i-1,j,-1] + pn[i,j+1,-1] + pn[i,j-1,-1] + pn[i,j,0] + pn[i,j,-2] - h*h*b[i,j,-1]) 

    return p
"""

#Solving Poisson equation with given function b of vector p, using cylindrical geometry. Returns the solution vector of the equation
def poisson(poiss,b,mask): 
    pn = np.empty_like(poiss)
    for q in range(nit):
        pn = poiss.copy()
        poiss[mask] = (1.0/6.0)*(np.roll(pn,1,0)[mask] + np.roll(pn,-1,0)[mask] + np.roll(pn,1,1)[mask] + np.roll(pn,-1,1)[mask] + np.roll(pn,1,2)[mask] + np.roll(pn,-1,2)[mask] - h*h*b[mask])  
    return poiss


#Computing psi function. Returns vector psi, and laplatian of vector psi (vector)
def compute_psi(phi0,lap_phi,mask):
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
def get_area2(phi,mask): #This definition not works very well
    return (np.absolute(phi[mask]) < 0.9).sum() 


#Computing Area. phi > 0 is a boolean array, so `sum` counts the number of True elements
def get_area3(phi,mask): 
    return ((2.0**0.5)*3.0/(eps*16.0))*(((phi[mask]**2-unity[mask])**2).sum()) #1 is a matrix with 1 - multiplying two matrices is not dot product! is just multiply number per number


#Computing Volume. phi > 0 is a boolean array, so `sum` counts the number of True elements
def get_volume(phi,mask):
    return ((((booll2>0.0).sum() + (phi[mask]).sum())/2))


#Lagrange Multiplier 1
def compute_multiplier_1(A0,Area): 
    return  alpha1 * (Area - A0)
#Lagrange Multiplier 1
def compute_multiplier_2(A0,Area): 
    return  alpha2 * (Area - A0)


#Lagrange Multiplier 2
def compute_multiplier_volume_before(V0,Volume):
    return  betabef * (Volume - (0.6*V0))
    #return  beta * (Volume - V0)

#Lagrange Multiplier 2
def compute_multiplier_volume(V0,Volume):
    return  beta * (Volume - V0)
    #return  beta * (Volume - V0)    


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

#We could study the velocity of the centre of mass

#----------------------------------------  Data print - Plot energy volume area  ----------------------------------------

#For printing all data. Plot energy, area, volume. I don't know the difference between areat i areat2 jeje
def dump(t,start):
 
    A_i0mod = ((A_x - A_x0)*(A_x - A_x0) + (A_y - A_y0)*(A_y - A_y0) + (A_z - A_z0)*(A_z - A_z0))
    A_xprov = A_x - A_x0
    A_yprov = A_y - A_y0
    A_zprov = A_z - A_z0
    w_xprov = w_x - w_x0
    w_yprov = w_y - w_y0
    w_zprov = w_z - w_z0
    w_i0mod = ((w_x - w_x0)*(w_x - w_x0) + (w_y - w_y0)*(w_y - w_y0) + (w_z - w_z0)*(w_z - w_z0))
    Area=get_area(phi,maskT)
    Area2=get_area3(phi,maskT)
    AltArea=get_area2(phi,maskT)
    Areatot= Area+Area2
    print(Areatot)
    volume = get_volume(phi,maskvol)
    energyb.append(np.sum((kappa*3*(2**0.5)/(8.0*(eps**3)))*(psi_sc[maskT]**2)))
    energym.append((np.sum((kappa*3*(2**0.5)/(8.0*(eps**3)))*(psi_sc[maskT]**2))) + Sigma2*Areatot)
    vol.append(volume)
    xi.append(A_i0mod.sum())
    omega.append(w_i0mod.sum())
    xi_x.append(((A_xprov)**2).sum())
    omega_x.append(((w_xprov)**2).sum())
    xi_y.append(((A_yprov)**2).sum())
    omega_y.append(((w_yprov)**2).sum())
    xi_z.append(((A_zprov)**2).sum())
    omega_z.append(((w_zprov)**2).sum())
    areat.append(Area)
    areat2.append(Area2)
    areatot.append(Areatot)
    areatalt.append(AltArea)
    volred.append((volume*6.0*(np.pi**0.5))/(Areatot**1.5))
    sav.append(Areatot/volume)
    now = datetime.now()
    print("Time t="+str(float(int(t/Tend*1000))/10)+" and took "+str(1/100*float(int(100*(time.perf_counter() - start)/1)))+"s - "+str(now.strftime("%d/%m/%Y %H:%M:%S"))) #Output of the system state at the given time t

    file1=os.path.join('./'+simulation+'/vorticity_x/vorticity_x_t='+str(t)+'.txt')
    file2=os.path.join('./'+simulation+'/vorticity_y/vorticity_y_t='+str(t)+'.txt')
    file3=os.path.join('./'+simulation+'/vorticity_z/vorticity_z_t='+str(t)+'.txt')
    file4=os.path.join('./'+simulation+'/vorticity_mod/vorticity_mod_t='+str(t)+'.txt')
    file5=os.path.join('./'+simulation+'/velocity_x/velocity_x_t='+str(t)+'.txt')
    file6=os.path.join('./'+simulation+'/velocity_y/velocity_y_t='+str(t)+'.txt')
    file7=os.path.join('./'+simulation+'/velocity_z/velocity_z_t='+str(t)+'.txt')
    file8=os.path.join('./'+simulation+'/velocity_mod/velocity_mod_t='+str(t)+'.txt')
    file9=os.path.join('./'+simulation+'/A_x/A_x_t='+str(t)+'.txt')
    file10=os.path.join('./'+simulation+'/A_y/A_y_t='+str(t)+'.txt')
    file11=os.path.join('./'+simulation+'/A_z/A_z_t='+str(t)+'.txt')
    file12=os.path.join('./'+simulation+'/A_mod/A_mod_t='+str(t)+'.txt')
    file15=os.path.join("./"+simulation+"/phi/phi_t="+str(t)+'.txt')
    with open(file1,'w+') as f1:
        with open(file2,'w+') as f2:
            with open(file3,'w+') as f3:
                with open(file4,'w+') as f4:
                    with open(file5,'w+') as f5:
                        with open(file6,'w+') as f6:
                            with open(file7,'w+') as f7:
                                with open(file8,'w+') as f8:
                                    with open(file9,'w+') as f9:
                                        with open(file10,'w+') as f10:
                                            with open(file11,'w+') as f11:
                                                with open(file12,'w+') as f12:
                                                    with open(file15,'w+') as f15:
                                                        for i in range(0,Nx):
                                                            for j in range(0,Ny):
                                                                for k in range(0,Nz):
                                                                    f1.write('{0} {1} {2} {3}\n'.format(i,j,k,w_x[i,j,k]))
                                                                    f2.write('{0} {1} {2} {3}\n'.format(i,j,k,w_y[i,j,k]))
                                                                    f3.write('{0} {1} {2} {3}\n'.format(i,j,k,w_z[i,j,k]))
                                                                    f4.write('{0} {1} {2} {3}\n'.format(i,j,k,w_mod[i,j,k]))
                                                                    f5.write('{0} {1} {2} {3}\n'.format(i,j,k,v_x[i,j,k]))
                                                                    f6.write('{0} {1} {2} {3}\n'.format(i,j,k,v_y[i,j,k]))
                                                                    f7.write('{0} {1} {2} {3}\n'.format(i,j,k,v_z[i,j,k]))
                                                                    f8.write('{0} {1} {2} {3}\n'.format(i,j,k,v_mod[i,j,k]))
                                                                    f9.write('{0} {1} {2} {3}\n'.format(i,j,k,A_x[i,j,k]))
                                                                    f10.write('{0} {1} {2} {3}\n'.format(i,j,k,A_y[i,j,k]))
                                                                    f11.write('{0} {1} {2} {3}\n'.format(i,j,k,A_z[i,j,k]))
                                                                    f12.write('{0} {1} {2} {3}\n'.format(i,j,k,A_mod[i,j,k]))
                                                                    f15.write("{0} {1} {2} {3}\n".format(i,j,k,phi[i,j,k]))

        file=os.path.join("./"+simulation+"/Energies-vol-area.txt")
        with open(file,"w+") as f:
            f.write("t bendenergy[t] membrenergy[t] vol[t] areat[t] areat2[t] alternarea[t] volred[t] areatot[t]\n")
            for t in range(0,len(energyb)):  
                f.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(t*tdump,energyb[t],energym[t],vol[t],areat[t],areat2[t],areatalt[t],volred[t],areatot[t],sav[t]))
            f.close()
            
        file=os.path.join("./"+simulation+"/OmegaXi.txt")
        with open(file,"w+") as f:
            f.write("t xi[t] omega[t] xi_x[t] omega_x[t] xi_y[t] omega_y[t] xi_z[t] omega_z[t]\n")
            for t in range(0,len(energyb)):  
                f.write("{0} {1} {2} {3} {4} {5} {6} {7} {8}\n".format(t*tdump,xi[t],omega[t],xi_x[t],omega_x[t],xi_y[t],omega_y[t],xi_z[t],omega_z[t]))
            f.close()


        #reduced volume plot
        plt.subplot(2,2,1)
        plt.plot(phi[Nx0,Ny0,:],'ro')
        plt.xlabel(r"${Length \, Channel}$",fontsize=12)
        plt.ylabel(r"${Phi\,(z)}$",fontsize=13)

        #volume plot
        plt.subplot(2,2,2)
        plt.plot(areatalt[1:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${Altarea\,(t)}$",fontsize=13)

        #Energy plot   
        plt.subplot(2,2,3)
        plt.plot(energyb[1:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${Bending\, energy\,(t)}$",fontsize=13)

        plt.subplot(2,2,4)
        plt.plot(energym[1:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${Membrane\, energy\,(t)}$",fontsize=13)

        figure = plt.gcf()  
        figure.set_size_inches(9, 9,forward=True)
        plt.tight_layout()
        file3=os.path.join("./"+simulation+"/energy.png")
        figure.savefig(file3, dpi = 200)
        
        plt.close('all')


        #Plot
        plt.subplot(2,2,1)
        plt.plot(areat[:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${areat\,(t)}$",fontsize=13)
        #plt.yscale('log')

        #Plot
        plt.subplot(2,2,2)
        plt.plot(areat2[:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${areat2\,(t)}$",fontsize=13)

        #Area2 plot
        plt.subplot(2,2,3)
        plt.plot(areatot[:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${areatot\,(t)}$",fontsize=13)

        #Volume plot
        plt.subplot(2,2,4)
        plt.plot(areatalt[:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${areatalt\,(t)}$",fontsize=13)

        figure = plt.gcf()  
        figure.set_size_inches(9, 9,forward=True)
        plt.tight_layout()
        file3=os.path.join("./"+simulation+"/Areat.png")
        figure.savefig(file3, dpi = 200)

        plt.close('all')

                #Plot
        plt.subplot(2,2,1)
        plt.plot(vol[:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${Volume\,(t)}$",fontsize=13)
        #plt.yscale('log')

        #Plot
        plt.subplot(2,2,2)
        plt.plot(areatot[:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${Area\,(t)}$",fontsize=13)

        #Area2 plot
        plt.subplot(2,2,3)
        plt.plot(volred[:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${RedVol\,(t)}$",fontsize=13)

        #Volume plot
        plt.subplot(2,2,4)
        plt.plot(sav[:])
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${SA-V\,(t)}$",fontsize=13)

        figure = plt.gcf()  
        figure.set_size_inches(9, 9,forward=True)
        plt.tight_layout()
        file3=os.path.join("./"+simulation+"/Area+volume.png")
        figure.savefig(file3, dpi = 200)

        plt.close('all')

        f, axarr = plt.subplots(1,1) 

        minmax=np.amax(phi[:,int(Ny*0.5),:]) 
        phiplot=axarr.imshow(phi[:,int(Ny*0.5),:],cmap='seismic',origin="lower", vmin = -minmax, vmax = minmax ) 
        f.colorbar(phiplot, ax=axarr, ticks=[-minmax*2/3,0,minmax*2/3],fraction=0.035, orientation='horizontal',format=FormatStrFormatter('%.3f'))

        figure = plt.gcf()
        figure.set_size_inches(12.0, 12.0,forward=True)
        #plt.tight_layout()
        file=os.path.join("./"+simulation+"/fig_phi/phit="+str(t)+'.png')
        figure.savefig(file, dpi = 200)
        plt.close('all')
        
        plt.subplot(2,1,1)
        plt.plot(xi)
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${\chi}$",fontsize=13)

        plt.subplot(2,1,2)
        plt.plot(omega)
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${\omega}$",fontsize=13)

        figure = plt.gcf()  
        figure.set_size_inches(9, 9,forward=True)
        plt.tight_layout()
        file3=os.path.join("./"+simulation+"/xiomegaplot/xiomega.png")
        figure.savefig(file3, dpi = 200)
        
        plt.close('all')
        
        
        plt.subplot(2,1,1)
        plt.plot(xi_x)
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${\chi_x}$",fontsize=13)

        plt.subplot(2,1,2)
        plt.plot(omega_x)
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${\omega_x}$",fontsize=13)

        figure = plt.gcf()  
        figure.set_size_inches(9, 9,forward=True)
        plt.tight_layout()
        file3=os.path.join("./"+simulation+"/xiomegaplot/xiomega_x.png")
        figure.savefig(file3, dpi = 200)
        
        plt.close('all')
        
        plt.subplot(2,1,1)
        plt.plot(xi_y)
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${\chi_y}$",fontsize=13)

        plt.subplot(2,1,2)
        plt.plot(omega_y)
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${\omega_y}$",fontsize=13)

        figure = plt.gcf()  
        figure.set_size_inches(9, 9,forward=True)
        plt.tight_layout()
        file3=os.path.join("./"+simulation+"/xiomegaplot/xiomega_y.png")
        figure.savefig(file3, dpi = 200)
        
        plt.close('all')
        
        plt.subplot(2,1,1)
        plt.plot(xi_z)
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${\chi_z}$",fontsize=13)

        plt.subplot(2,1,2)
        plt.plot(omega_z)
        plt.xlabel(str(int(tdump))+r"$\,\Delta t$",fontsize=12)
        plt.ylabel(r"${\omega_z}$",fontsize=13)

        figure = plt.gcf()  
        figure.set_size_inches(9, 9,forward=True)
        plt.tight_layout()
        file3=os.path.join("./"+simulation+"/xiomegaplot/xiomega_z.png")
        figure.savefig(file3, dpi = 200)
        
        plt.close('all')

        
        f, axarr = plt.subplots(2,2)

        
        wyrestaplot=axarr[0,0].imshow(w_yprov[:,int((Ny-1)/2),:],cmap='seismic',origin="lower")
        f.colorbar(wyrestaplot, ax=axarr[0,0], orientation="horizontal")

        phiplot1=axarr[0,1].imshow(phi[:,int((Ny-1)/2),:],cmap='seismic',origin="lower" )
        f.colorbar(phiplot1, ax=axarr[0,1],orientation="horizontal")

        Ayrestaplot=axarr[1,0].imshow(A_yprov[:,int((Ny-1)/2),:],cmap='seismic',origin="lower")
        f.colorbar(Ayrestaplot, ax=axarr[1,0], orientation="horizontal")

        phiplot2=axarr[1,1].imshow(phi[:,int((Ny-1)/2),:],cmap='seismic',origin="lower" )
        f.colorbar(phiplot2, ax=axarr[1,1],orientation="horizontal")

        axarr[0,0].set_title('w_y - wy_0')
        axarr[1,0].set_title('A_y - Ay_0')
        axarr[0,1].set_title('Phi')
        axarr[1,1].set_title('Phi')
        
        figure = plt.gcf()
        figure.set_size_inches(4.5, 4.5,forward=True)
        plt.tight_layout()
        file=os.path.join("./"+simulation+"/xiomega_y/plotAwyt="+str(t)+'.png')
        figure.savefig(file, dpi = 200)
        plt.close('all')
    
#----------------------------------------  Creating folders  ----------------------------------------

print(os.getcwd())
  
if not os.path.exists("./"+simulation+"/"):
    os.makedirs("./"+simulation+"/")
    print("new folder "+simulation)
if not os.path.exists("./"+simulation+"/vorticity_x/"):
    os.makedirs("./"+simulation+"/vorticity_x/")
    print("new folder vorticity_x")
if not os.path.exists("./"+simulation+"/A_x/"):
    os.makedirs("./"+simulation+"/A_x/")
    print("new folder A_x")
if not os.path.exists("./"+simulation+"/velocity_x/"):
    os.makedirs("./"+simulation+"/velocity_x/")
    print("new folder velocity_x")
if not os.path.exists("./"+simulation+"/vorticity_y/"):
    os.makedirs("./"+simulation+"/vorticity_y/")
    print("new folder vorticity_y")
if not os.path.exists("./"+simulation+"/A_y/"):
    os.makedirs("./"+simulation+"/A_y/")
    print("new folder A_y")
if not os.path.exists("./"+simulation+"/velocity_y/"):
    os.makedirs("./"+simulation+"/velocity_y/")
    print("new folder velocity_y")
if not os.path.exists("./"+simulation+"/A_z/"):
    os.makedirs("./"+simulation+"/A_z/")
    print("new folder A_z")
if not os.path.exists("./"+simulation+"/vorticity_z/"):
    os.makedirs("./"+simulation+"/vorticity_z/")
    print("new folder vorticity_z")
if not os.path.exists("./"+simulation+"/velocity_z/"):
    os.makedirs("./"+simulation+"/velocity_z/")
    print("new folder velocity_z")
if not os.path.exists("./"+simulation+"/vorticity_mod/"):
    os.makedirs("./"+simulation+"/vorticity_mod/")
    print("new folder vorticity_mod")
if not os.path.exists("./"+simulation+"/velocity_mod/"):
    os.makedirs("./"+simulation+"/velocity_mod/")
    print("new folder velocity_mod")
if not os.path.exists("./"+simulation+"/A_mod/"):
    os.makedirs("./"+simulation+"/A_mod/")
    print("new folder A_mod")
if not os.path.exists("./"+simulation+"/phi/"):
    os.makedirs("./"+simulation+"/phi/")
    print("new folder phi")
if not os.path.exists("./"+simulation+"/fig_phi/"):
    os.makedirs("./"+simulation+"/fig_phi/")
    print("new folder fig_phi")
output1=os.path.join("./"+simulation+"/xiomegaplot/")
if not os.path.exists(output1):
    os.makedirs(output1)
    print("new folder xiomegaplot")
output1=os.path.join("./"+simulation+"/vti/")
if not os.path.exists(output1):
    os.makedirs(output1)
    print("new folder vti")
output1=os.path.join("./"+simulation+"/xiomega/")
if not os.path.exists(output1):
    os.makedirs(output1)
    print("new folder xiomega")
    
output1=os.path.join("./"+simulation+"/xiomega_y/")
if not os.path.exists(output1):
    os.makedirs(output1)
    print("new folder xiomega_y")

dodo=os.path.join("./"+simulation+"/"+os.path.basename(__file__))
copyfile(__file__,dodo)  


#----------------------------------------  Main  ----------------------------------------


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
w_xprov = np.zeros((Nx,Ny,Nz),float)
w_yprov = np.zeros((Nx,Ny,Nz),float)
w_zprov = np.zeros((Nx,Ny,Nz),float)
A_x0 = np.zeros((Nx,Ny,Nz),float)
A_xprov = np.zeros((Nx,Ny,Nz),float)
A_yprov = np.zeros((Nx,Ny,Nz),float)
A_zprov = np.zeros((Nx,Ny,Nz),float)
A_y0 = np.zeros((Nx,Ny,Nz),float)
A_z0 = np.zeros((Nx,Ny,Nz),float)
A_i0mod = np.zeros((Nx,Ny,Nz),float)
w_i0mod = np.zeros((Nx,Ny,Nz),float)
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
R1=14.0
R2=0.29*R1
sigmavmatrix = np.zeros_like(phi)


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
                #rot_i = do_rotation_yz_x(i,j,k)
                #rot_j = do_rotation_yz_y(i,j,k)
                #rot_k = do_rotation_yz_z(i,j,k)
                if ((float(i)-Nx0)*(float(i)-Nx0)/(R1**2)+(float(j)-Ny0)*(float(j)-Ny0)/(R1**2)+(float(k)-Nz0)*(float(k)-Nz0)/(R2**2)>1.0): # elipsoidal initial conditions
                #if ((float(i)-Nx0)*(float(i)-Nx0)+(float(j)-Ny0)*(float(j)-Ny0)+(float(k)-Nz0)*(float(k)-Nz0)>R1*R1): # sphereinitial conditions
                #if ((rot_i)*(rot_i)/(R1**2)+(rot_j)*(rot_j)/(R1**2)+(rot_k)*(rot_k)/(R2**2)>1.0): # rotated elipsoidal initial conditions
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
#volini=(4.0/3.0)*0.29*np.pi*(R1**3)
#areaini= (4.0*np.pi*(R1**2)*0.5868)
#volini=(4.0/3.0)*np.pi*(R1**3)
#areaini= (4.0*np.pi*(R1**2))
#areainiasaco = 0.0
#for i in range(0,Nx):
 #   for j in range(0, Ny):
  #      for z in range(0, Nz):
   #         if(phi[i][j][z] > 0.0):
    #            if(phi[i+1][j][z] < 0.0 or phi[i-1][j][z] < 0.0 or phi[i][j+1][z] < 0.0 or phi[i][j-1][z] < 0.0 or phi[i][j][z+1] < 0.0 or phi[i][j][z-1] < 0.0):
     #               areainiasaco = areainiasaco + 1.0
#volredini = (volini*6.0*(np.pi**0.5))/(areaini**1.5)
#savini = (areaini/volini)
maskT=newbooll2>0.5
maskvol = booll2>0.5


#print("Initial areaasaco -> ", areainiasaco)

start = time.perf_counter()

Area=get_area(phi,maskT)
Area2=get_area3(phi,maskT)
AltArea=get_area2(phi,maskT)
Areatot= Area+Area2
volume = get_volume(phi,maskvol)
area2 = 0.0
area3 = 0.0
volini=(4.0/3.0)*np.pi*(R1**3)

print("Initial compuarea -> ", Areatot)
print("Initial compuareaalt -> ", AltArea)
print("Initial compuvol -> ", volume)
print("Initial volini -> ", volini)


for t in range(0,t1):
    lap_phi = lap_roll(phi,maskT)
    lap_lap_phi=lap_roll(lap_phi,maskT)
    psi_sc,lap_psi = compute_psi(phi,lap_phi,maskT)
    mu_b,lap_mu_b = compute_mu_b(phi,psi_sc,lap_psi,maskT)

    if(t < t0):
        phi[maskT] += dt*M*(lap_mu_b[maskT])
    
    elif(t == t0):  
        A01= get_area(phi,maskT)
        A02= get_area3(phi,maskT)
        V0=get_volume(phi,maskvol)
        phi[maskT] += dt*M*(lap_mu_b[maskT])

    elif(t > t0 and t < t1):
        v=get_volume(phi,maskvol)
        area1=get_area(phi,maskT)
        area3=get_area3(phi,maskT)
        grad_lapmu_b = grad_roll(lap_mu_b,maskT)
        #Sigma = compute_multiplier_1(A01,area1)
        Sigma2 = compute_multiplier_2(A01+A02,area1+area3)
        #Sigma3 = compute_multiplier_volume_before(V0,v)
        Sigma3 = compute_multiplier_volume(V0,v)
        sigmavmatrix = Sigma3*unity
        mu_multipl,lap_mu_multipl = compute_mu_multipl(phi, lap_phi, Sigma, Sigma2, Sigma3, maskT)
        phi[maskT] += dt*M*(lap_mu_b[maskT]+lap_mu_multipl[maskT]+sigmavmatrix[maskT])

    #Print data- Only phi changes: velocity vorticity and A remain constant
    if (t % tdump == 0): 
        v_mod = (v_x*v_x + v_y*v_y + v_z*v_z)**(0.5)
        w_mod = (w_x*w_x + w_y*w_y + w_z*w_z)**(0.5)
        A_mod = (A_x*A_x + A_y*A_y + A_z*A_z)**(0.5)
        A_i0mod = ((A_x - A_x0)*(A_x - A_x0) + (A_y - A_y0)*(A_y - A_y0) + (A_z - A_z0)*(A_z - A_z0))
        A_xprov = A_x - A_x0
        A_yprov = A_y - A_y0
        A_zprov = A_z - A_z0
        w_xprov = w_x - w_x0
        w_yprov = w_y - w_y0
        w_zprov = w_z - w_z0
        w_i0mod = ((w_x - w_x0)*(w_x - w_x0) + (w_y - w_y0)*(w_y - w_y0) + (w_z - w_z0)*(w_z - w_z0))
        A_i0modsq = np.empty_like(A_i0mod)
        A_i0modsq = (A_i0mod)**(0.5)
        w_i0modsq = np.empty_like(w_i0mod)
        w_i0modsq = (w_i0mod)**(0.5)

        dump(t,start)
        save_vti_file(phi,v_x,v_y,v_z, Nx, Ny, Nz)
        start = time.perf_counter()        
        

#--------------------  Main temporal loop  --------------------

#aux1=get_area3(phi,maskT)
#aux2=get_area(phi,maskT)
#A02= aux1+aux2
#V02=get_volume(phi,maskvol)
for t in range(t1,Tend):

    #0-Calculating parameters
    maskphi = phi>0.0
    lap_phi = lap_roll(phi,maskT)
    psi_sc,lap_psi = compute_psi(phi,lap_phi,maskT)
    #viscosity = compute_visco(phi,maskT) #just if the contrast is =! 1
    area1=get_area(phi,maskT)
    area3=get_area3(phi,maskT)
    v=get_volume(phi,maskvol) 
    #Sigma = compute_multiplier_1(A01,area1)
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
        A_i0mod = ((A_x - A_x0)*(A_x - A_x0) + (A_y - A_y0)*(A_y - A_y0) + (A_z - A_z0)*(A_z - A_z0))
        A_xprov = A_x - A_x0
        A_yprov = A_y - A_y0
        A_zprov = A_z - A_z0
        w_xprov = w_x - w_x0
        w_yprov = w_y - w_y0
        w_zprov = w_z - w_z0
        w_i0mod = ((w_x - w_x0)*(w_x - w_x0) + (w_y - w_y0)*(w_y - w_y0) + (w_z - w_z0)*(w_z - w_z0))
        A_i0modsq = np.empty_like(A_i0mod)
        A_i0modsq = (A_i0mod)**(0.5)
        w_i0modsq = np.empty_like(w_i0mod)
        w_i0modsq = (w_i0mod)**(0.5)

        dump(t,start)
        save_vti_file(phi,v_x,v_y,v_z, Nx, Ny, Nz)
        start = time.perf_counter()


