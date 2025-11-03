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

def output_folders(simulation):
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
    return


#----------------------------------------  Data print - Plot energy volume area  ----------------------------------------
#For printing all data. Plot energy, area, volume. I don't know the difference between areat i areat2 jeje
def dump(t,start,Tend, tdump, simulation, energyb, energym,h,eps,unity, booll2, kappa,
    maskT,maskvol,phi,xi, omega, xi_x, xi_y,xi_z,omega_x,omega_y,omega_z,
    vol, areat, areat2, areatot, areatalt, volred, sav, psi_sc,Sigma2,
    Nx,Ny,Nz,Nx0,Ny0,Nz0,w_mod, A_mod, v_x,v_y,v_z,v_mod,
    A_x,A_y,A_z,A_x0,A_y0,A_z0,w_x,w_y,w_z,w_x0,w_y0,w_z0): 

    A_i0mod = ((A_x - A_x0)*(A_x - A_x0) + (A_y - A_y0)*(A_y - A_y0) + (A_z - A_z0)*(A_z - A_z0))
    A_xprov = A_x - A_x0
    A_yprov = A_y - A_y0
    A_zprov = A_z - A_z0
    w_xprov = w_x - w_x0
    w_yprov = w_y - w_y0
    w_zprov = w_z - w_z0
    w_i0mod = ((w_x - w_x0)*(w_x - w_x0) + (w_y - w_y0)*(w_y - w_y0) + (w_z - w_z0)*(w_z - w_z0))
    Area=get_area(phi,maskT,eps,h)
    Area2=get_area3(phi,maskT,eps,unity)
    AltArea=get_area2(phi,maskT)
    Areatot= Area+Area2
    print(Areatot)
    volume = get_volume(phi,maskvol,booll2)
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
    #Output of the system state at the given time t
    print("Time t="+str(float(int(t/Tend*1000))/10)+" and took "+str(1/100*float(int(100*(time.perf_counter() - start)/1)))+"s - "+str(now.strftime("%d/%m/%Y %H:%M:%S"))) 

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
    
def save_vti_file(t,phi,vx,vy,vz, nx, ny, nz,booll2,newbooll2,simulation):
    
    pc_lista_novo_phi = [] 
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



# the bulk in the system has gradient 0 therefore only the membrane contributes
def get_area(phi,mask,eps,h):
    aux1 = grad_roll(phi,mask,h)
    return ((2.0**0.5)*3.0*eps/8.0)*(np.sum(aux1[0][mask]*aux1[0][mask]+aux1[1][mask]*aux1[1][mask]+aux1[2][mask]*aux1[2][mask]))

#Computing Area. phi > 0 is a boolean array, so `sum` counts the number of True elements
def get_area2(phi,mask): #This definition not works very well
    return (np.absolute(phi[mask]) < 0.9).sum() 

#Computing Area. phi > 0 is a boolean array, so `sum` counts the number of True elements
def get_area3(phi,mask,eps,unity): 
    return ((2.0**0.5)*3.0/(eps*16.0))*(((phi[mask]**2-unity[mask])**2).sum()) #1 is a matrix with 1 - multiplying two matrices is not dot product! is just multiply number per number
 

def grad_roll(u,mask,h):  #gradient using roll of a given field u. Returns a vector of 3 vectors [0][1][2]
    gradux = np.zeros_like(u)
    graduy = np.zeros_like(u)
    graduz = np.zeros_like(u)
    gradux[mask] = (0.5/h)*(-np.roll(u,1,0)[mask]+np.roll(u,-1,0)[mask])
    graduy[mask] = (0.5/h)*(-np.roll(u,1,1)[mask]+np.roll(u,-1,1)[mask])
    graduz[mask] = (0.5/h)*(-np.roll(u,1,2)[mask]+np.roll(u,-1,2)[mask])
    gu = [gradux,graduy,graduz]
    return gu
    
#Computing Volume. phi > 0 is a boolean array, so `sum` counts the number of True elements
def get_volume(phi,mask,booll2):
    return ((((booll2>0.0).sum() + (phi[mask]).sum())/2))

    #Computing psi function. Returns vector psi, and laplatian of vector psi (vector)
def compute_psi(phi0,lap_phi,mask,unity,eps,C0):
    psi_sc = np.zeros_like(phi0)
    psi_sc[mask]=phi0[mask]*(-unity[mask]+phi0[mask]*phi0[mask])-eps*eps*lap_phi[mask]-eps*C0*(unity[mask]-phi0[mask]*phi0[mask])
    lap_psi= lap_roll(psi_sc,mask)
    return psi_sc,lap_psi
