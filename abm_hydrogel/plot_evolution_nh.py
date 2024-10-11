#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:10:19 2023

@authors:    André H. Erhardt (andre.erhardt@wias-berlin.de, https://orcid.org/0000-0003-4389-8554)
             Dirk Peschka (dirk.peschka@wias-berlin.de, https://orcid.org/0000-0002-3047-1140)

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['font.size'] = '12'
plt.rcParams["font.family"] = "Times New Roman"
    
def extend_F(F,J):
    detF = F[0,0]*F[1,1]-F[0,1]*F[1,0]
    F33 = J / detF
    return as_tensor([[F[0,0],F[0,1],0],[F[1,0],F[1,1],0],[0,0,F33]])  

def plot_agents(p1l,p2l,u_displacement):
    npoint = np.size(p1l,0)
    for i in range(0,npoint):
        p1 = Point(p1l[i][0],p1l[i][1])
        p2 = Point(p2l[i][0],p2l[i][1])
        p_set1 = p1 + Point(u_displacement(p1))
        p_set2 = p2 + Point(u_displacement(p2))
        plt.plot((p_set1[0],p_set2[0]),(p_set1[1],p_set2[1]),color='r',linewidth=1,zorder=2)


def plot_barrier(Q, MESH, CASE,P1,P2,i,j,CONC=False,XLABEL=True,YLABEL=True,fMIN=0.0,fMAX=2.0):
    
    fsteps = 256

    if not(CONC):
        mymap = cm.jet
    else:
        mymap = cm.bwr_r 
    
    q    = Q[i]
    case = CASE[i]
    mesh = MESH[i]
    if planestrain == True:
        if case == 3 or case == 4:
            u, p, eta, psi  = split(q)
        else:
            u, p            = split(q)
            psi = Constant(phi0)
    else:
        if case == 3 or case == 4:
            u, eta, psi     = split(q)
        else:
            u               = q
            psi = Constant(phi0)

    meshc = Mesh(mesh)
    u = project(u,VectorFunctionSpace(meshc,'CG',2))    

    I   = Identity(2)
    F   = I + grad(u)
    I3  = Identity(3)

    if case == 3 or case == 4:
        J_bar = (1+alpha0*(psi-phi0))
    else:
        J_bar = 1

    F_bar   = extend_F(F,J_bar)
    C       = F_bar.T*F_bar 
    
    xmeshL = RectangleMesh(Point(0,0),Point(0.001,1),1,100,"left/right")
    xmeshR = RectangleMesh(Point(0.999,0),Point(1,1),1,100,"left/right")
    WL     = VectorFunctionSpace(xmeshL, 'CG' ,degree = 2)  
    WR     = VectorFunctionSpace(xmeshR, 'CG' ,degree = 2)  
    uL = project(u,WL)
    uR = project(u,WR)

    if not(CONC):
        plotfunc = project(tr(C-I3),FunctionSpace(meshc,'CG',2))
    else:
        plotfunc = project(psi,FunctionSpace(meshc,'CG',2))

    maximum = plotfunc.vector().max()
    minimum = plotfunc.vector().min()

    print("i=",i," min=",minimum," max=",maximum)

    # plt.subplot(NX,NY,j)
    ax = plt.subplot(NX,NY,j)

    du = u(0.5,1)
    plt.plot((0,1),(1+du[1],1+du[1]),linewidth=1,color='k',linestyle = 'dotted')

    plot_agents(P1[i],P2[i],u)

    ALE.move(meshc , u)
    ALE.move(xmeshL, uL)
    ALE.move(xmeshR, uR)

    c = plot(plotfunc, cmap=mymap,vmin=fMIN,vmax=fMAX,levels = np.linspace(fMIN, fMAX, fsteps),extend='both',zorder=1)

    plot(xmeshL,linewidth= 1,color='k')
    plot(xmeshR,linewidth= 1,color='k')
    
    
    if not(YLABEL):
        plt.gca().set_yticks([])
    else:
        plt.gca().set_yticks([0,0.5,1,1.5,2])
    if not(XLABEL):
        plt.gca().set_xticks([])
    else:
        plt.gca().set_xticks([0,0.5,1])
    
    plt.xlim([-0.05,1.05])
    plt.ylim([0,2.3])

    cbaxes = inset_axes(ax, width="50%", height="3%", loc='upper center') 
    m = plt.cm.ScalarMappable(cmap=mymap)
    m.set_array(plotfunc.vector())
    m.set_clim(fMIN, fMAX)
    gcl = plt.colorbar(m, cax=cbaxes,boundaries=np.linspace(fMIN, fMAX, fsteps),ticks=[fMIN,fMAX],orientation='horizontal')
        
    gcl.ax.tick_params(labelsize=12)
        

folders = ['output_hydro_NH_stretched_AC_05/'] 
kk      = 1
l       = []
CASE    = []
Q       = []
MESH    = []
N1      = []
N2      = []

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
    
    R = [61,511,1021,3000]

    abm,mesh,q,W = load_complete_state(output_folder + 'initial',abm_pars)
    (n1list,n2list,clist,node1_x,node1_y,node2_x,node2_y,ndipole) = abm.get_dipole_direction_point_source()
        
    CASE.append(case)
    Q.append(q)
    MESH.append(mesh)
    N1.append(n1list)
    N2.append(n2list)

    for i in R:
        abm,mesh,q,W = load_complete_state(output_folder + 'state'+str(i),abm_pars)
        (n1list,n2list,clist,node1_x,node1_y,node2_x,node2_y,ndipole) = abm.get_dipole_direction_point_source()
        
        CASE.append(case)
        Q.append(q)
        MESH.append(mesh)
        N1.append(n1list)
        N2.append(n2list)

NY = 5
NX = 2

fig, ax = plt.subplots(NX,NY,figsize=(9,7), squeeze=False)

plot_barrier(Q , MESH , CASE , N1 , N2 , 0 , 1 , fMIN=-0.04 , fMAX=0.04                       ,XLABEL=False)
plot_barrier(Q , MESH , CASE , N1 , N2 , 1 , 2 , fMIN= 0.04 , fMAX=0.14          ,YLABEL=False,XLABEL=False)
plot_barrier(Q , MESH , CASE , N1 , N2 , 2 , 3 , fMIN= 1.0  , fMAX=1.3           ,YLABEL=False,XLABEL=False)
plot_barrier(Q , MESH , CASE , N1 , N2 , 3 , 4 , fMIN= 2.5  , fMAX=2.9           ,YLABEL=False,XLABEL=False)
plot_barrier(Q , MESH , CASE , N1 , N2 , 4 , 5 , fMIN= 2.5  , fMAX=2.9           ,YLABEL=False,XLABEL=False)
plot_barrier(Q , MESH , CASE , N1 , N2 , 0 , 6 , fMIN= 0.0  , fMAX=0.4 ,CONC=True                          )
plot_barrier(Q , MESH , CASE , N1 , N2 , 1 , 7 , fMIN= 0.0  , fMAX=0.4 ,CONC=True,YLABEL=False             )
plot_barrier(Q , MESH , CASE , N1 , N2 , 2 , 8 , fMIN= 0.0  , fMAX=0.4 ,CONC=True,YLABEL=False             )
plot_barrier(Q , MESH , CASE , N1 , N2 , 3 , 9 , fMIN= 0.0  , fMAX=0.4 ,CONC=True,YLABEL=False             )
plot_barrier(Q , MESH , CASE , N1 , N2 , 4 ,10 , fMIN= 0.0  , fMAX=0.4 ,CONC=True,YLABEL=False             )

plt.tight_layout()

plt.savefig('ABM_nh_stretching.pdf',dpi=250)
plt.savefig('ABM_nh_stretching.jpeg',dpi=250)
