#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:10:19 2023

@authors:    André H. Erhardt (andre.erhardt@wias-berlin.de, https://orcid.org/0000-0003-4389-8554)

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab
import sys
import os
from fenics import *
from abm_module import *
from os import mkdir

plt.rcParams['font.size'] = '10'
plt.rcParams["font.family"] = "Times New Roman"
    
def extend_F(F,J):
    detF = F[0,0]*F[1,1]-F[0,1]*F[1,0]
    F33 = J / detF
    return as_tensor([[F[0,0],F[0,1],0],[F[1,0],F[1,1],0],[0,0,F33]])  

def plot_barrier(Q, MESH, CASE,i,j,LABEL=False,BAR=False):
    
    MAX = []
    MIN = []
    fMIN = 0
    fMAX = 4
    fsteps = 100

    mymap     = cm.jet
    mymap_psi = cm.Blues

    mmin = (0.1,0.6,1.6,0.1,0.6,1.6)
    mmax = (0.4,1.0,2.0,0.4,1.0,2.0)
    mmin = (0.07,1.0,1.76,0.07,1.0,1.76)
    mmax = (0.17,1.4,2.00,0.17,1.4,2.00 )

    q    = Q[i]
    case = CASE[i]
    mesh = MESH[i]
    if planestrain == True:
        if case == 3 or case == 4:
            u, p, eta, psi  = split(q)
        else:
            u, p            = split(q)
    else:
        if case == 3 or case == 4:
            u, eta, psi     = split(q)
        else:
            u               = q
            
    I   = Identity(2)
    F   = I + grad(u)
    I3  = Identity(3)
    if case == 3 or case == 4:
        J_bar = (1+alpha0*(psi-phi0))
    else:
        J_bar = 1
    F_bar   = extend_F(F,J_bar)
    C       = F_bar.T*F_bar 
    barrier = project(tr(C-I3),FunctionSpace(mesh,'CG',2))
    
    xmeshL = RectangleMesh(Point(0,0),Point(0.005,1),1,100,"left/right")
    xmeshR = RectangleMesh(Point(0.995,0),Point(1,1),1,100,"left/right")
    WL     = VectorFunctionSpace(xmeshL, 'CG' ,degree = 2)  
    WR     = VectorFunctionSpace(xmeshR, 'CG' ,degree = 2)  

    uL = project(u,WL)
    uR = project(u,WR)
    ALE.move(mesh, u)
    ALE.move(xmeshL, uL)
    ALE.move(xmeshR, uR)
        
    maximum = barrier.vector().max()
    minimum = barrier.vector().min()
    print("i=",i," min=",minimum," max=",maximum)
    
    MAX.append(maximum)
    MIN.append(minimum)
    fMIN = mmin[i]
    fMAX = mmax[i] # maximum # (maximum - minimum)/2 + minimum
    
    # plt.subplot(1, 5, i); surf = plot(barrier, vmin = fMIN, vmax = fMAX, cmap=cm.jet)
    plt.subplot(3, 2,j); 
    # surf = plot(barrier, vmin = fMIN, vmax = fMAX, cmap=mymap, levels = np.linspace(fMIN, fMAX, fsteps))
    c = plot(barrier, cmap=mymap,vmin=fMIN,vmax=fMAX,levels = np.linspace(fMIN, fMAX, fsteps))
    plot(xmeshL,linewidth= 0.25,color='k')
    plot(xmeshR,linewidth= 0.25,color='k')
    # plt.axis('equal')
    #if not(XTICKS):
    #    plt.gca().set_xticks([])
    if BAR:
        plt.gca().set_yticks([])
        m = plt.cm.ScalarMappable(cmap=cm.jet)
        m.set_array(barrier.vector())
        m.set_clim(fMIN, fMAX)
        # m.set_ylabel('$\mathrm{tr}(\mathbf{F}^T\cdot \mathbf{F})-3$', loc='center')
        if LABEL:
            plt.colorbar(m, boundaries=np.linspace(fMIN, fMAX, fsteps),ticks=[fMIN,(fMIN+fMAX)/2,fMAX],label='$\mathrm{tr}(\mathbf{F}^T \mathbf{F}-\mathbb{I})$')
        else:
            plt.colorbar(m, boundaries=np.linspace(fMIN, fMAX, fsteps),ticks=[fMIN,(fMIN+fMAX)/2,fMAX])
    plt.xlim([-0.05,1.05])
    plt.ylim([0,2])
    # ax.set_ylabel(c, loc='center')
    #i += 1
    #plt.colorbar(c)
    #plt.xlim([-0.1,3.4])
    #plt.ylim([0,1.75])

    
    
    #plt.subplot(2,6,5)
    #m = plt.cm.ScalarMappable(cmap=mymap)
    #m.set_array(barrier.vector())
    #m.set_clim(fMIN, fMAX)
    #plt.colorbar(m, boundaries=np.linspace(fMIN, fMAX, fsteps))
    #plt.subplot(2,6,11)
    #m = plt.cm.ScalarMappable(cmap=mymap_psi)
    #m.set_array(psi.vector())
    #m.set_clim(fMIN_psi, fMAX_psi)
    #plt.colorbar(m, boundaries=np.linspace(fMIN_psi, fMAX_psi, fsteps))
    #cbar.ax.set_yticklabels([format(min(MIN),'.1e'), format(max(MAX),'.1e')],fontsize = 10) 
    #cbar.ax.set_ylabel('$\mathrm{tr}(\mathbf{F}^T\cdot \mathbf{F})-3$', loc='center')
    #ax[0,0].set_ylabel('y coordinate') # Y label
    #ax[0,2].set_xlabel('x coordinate') # X 
    # ax.tick_params(labelsize=20)
    #plt.savefig(outputfolder + file_name + 'barrier' + ".pdf",dpi=1200, bbox_inches='tight',pad_inches = 0) 
    
# folders = ['output_hydro_NH_CH_05/','output_hydro_gent_CH_05/','output_hydro_gent_AC_0005/','output_hydro_gent_AC_05/','output_hydro_gent_AC_50/'] 
# folders = ['output_elastic_gent/','output_elastic_NH/'] 
folders = ['output_elastic_gent/','output_elastic_NH/'] 
kk = 1
l = []
CASE = []
Q = []
MESH = []
    
for folder in folders:
    abm_pars = read_dictionary(folder + 'run_pars.json')
    # =========================================================================== #
    print("{:<15} {:<10}".format('Parameter','Value'))
    for v in abm_pars.items():
        label, num = v
        print("{:<15} {:<10}".format(label, num)) 
    # =========================================================================== #
    t_final       = abm_pars["t_final"]
    alpha         = abm_pars["alpha"]
    traction      = abm_pars["traction"]
    nu            = abm_pars["nu"]
    Im            = abm_pars["Im"]
    Jm            = abm_pars["Jm"]
    n_steps       = abm_pars["n_steps"]
    dt            = abm_pars["dt"]
    width         = abm_pars["width"]
    height        = abm_pars["height"]
    Ncell         = abm_pars["Ncell"]
    empty_ratio   = abm_pars["empty_ratio"]
    output_folder = abm_pars["output_folder"]
    mob           = abm_pars["mob"]
    Nmono         = abm_pars["Nmono"]
    alpha0        = abm_pars["alpha0"]
    eps           = abm_pars["eps"]
    phi0          = abm_pars["phi0"]
    C0            = abm_pars["C0"]
    planestrain   = abm_pars["planestrain"]
    case          = abm_pars["case"]
    AllenCahn     = abm_pars["AllenCahn"]

    if planestrain == True:
        C1 = 1/C0
    else:
        C1 = C0
    
    #for i in range(n_steps+1,1,-1):  
    i = 941
    #if os.path.exists(output_folder + 'state'+str(i)+ '.h5'):
    R = [201,701,951]
    for i in R:
        mesh,q,W = load_state(output_folder + 'state'+str(i),case,planestrain)
        CASE.append(case)
        Q.append(q)
        MESH.append(mesh)

fig, ax = plt.subplots(2,6,figsize=(3.9,8), squeeze=False)
    
plot_barrier(Q, MESH, CASE,0,1)
plot_barrier(Q, MESH, CASE,1,3)
plot_barrier(Q, MESH, CASE,2,5)
plot_barrier(Q, MESH, CASE,3,2,BAR=True)
plot_barrier(Q, MESH, CASE,4,4,BAR=True,LABEL=True)
plot_barrier(Q, MESH, CASE,5,6,BAR=True)

plt.tight_layout()

# plt.show()

plt.savefig('strainstiffening.pdf',dpi=300)