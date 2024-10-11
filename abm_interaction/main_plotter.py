#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:48:46 2023

@authors:    André H. Erhardt (andre.erhardt@wias-berlin.de, https://orcid.org/0000-0003-4389-8554)
             Dirk Peschka (dirk.peschka@wias-berlin.de, https://orcid.org/0000-0002-3047-1140)

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.cm as cm

import pylab
import sys
from fenics import *
from abm_module import *
from os import mkdir
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from sigfig import round

random.seed(30)

plt.rcParams['font.size'] = '17'
plt.rcParams["font.family"] = "Times New Roman"

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 16

# =========================================================================== #
# Read simulation parameters from json file and convert to global variables

if len(sys.argv) == 1:
    in_json = 'default.json'
else:
    in_json = sys.argv[1]

print("Reading ",in_json)
abm_pars = read_dictionary(in_json)

print("============================================")
print("{:<15} {:<10}".format('Parameter','Value'))
print("--------------------------------------------")
for v in abm_pars.items():
    label, num = v
    print("{:<15} {:<10}".format(label, num)) 
print("============================================")

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
NEWTON_SOLVER = abm_pars["NEWTON_SOLVER"]
fname         = abm_pars["name"]

C0RUN         = abm_pars["C0RUN"]
C1RUN         = abm_pars["C1RUN"]
C2RUN         = abm_pars["C2RUN"]


C1 = C0

try:
    mkdir(output_folder)
except:
    print("Folder already exists")

write_dictionary(output_folder + 'run_pars.json',abm_pars)

# =========================================================================== #
"""defining the agent-based problem""" 
t        = 0
mesh     = RectangleMesh(Point(0,0),Point(1,1),100,100,"left/right")
W        = get_space(mesh,case,planestrain)

"""Optimization options for the form compiler"""
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 5

"""initialisation of the agent-based model"""
# random.seed(2)  
# abm         = ABM(width, height,empty_ratio,traction)             
# W           = get_space(mesh,case,planestrain)
# q           = interpolate(initial, W)
# old_q       = interpolate(initial, W)
# upd         = project(correction,W)
# orient_name = ["vert  ","horiz ","diag 1","diag 2"]

output_folder = abm_pars["output_folder"]

# =========================================================================== #
def plot_line(u,xx,yy,hh,h=0.01,LW=1):


    orient = { 0 : [[0,0],[-h,h]], 
               1 : [[-h,h],[0,0]],
               2 : [[-h,h],[-h,h]],
               3 : [[-h,h],[h,-h]]
             }
    [dx,dy] = orient[hh]
    x1=xx + dx[0]
    x2=xx + dx[1]
    y1=yy + dy[0]
    y2=yy + dy[1]
    du1 = u(x1,y1)
    du2 = u(x2,y2)
    plt.plot([x1+du1[0],x2+du2[0]],[y1+du1[1],y2+du2[1]],'r-',linewidth=LW)

if C0RUN:
    
    mesh,q,W = load_state(output_folder + 'deformed_C0',case,planestrain)
    if planestrain == True:
        if case == 3 or case == 4:
            u, p, eta, psi = split(q)
        else:
    
            u, p           = split(q)
    else:
        if case == 3 or case == 4:
            u, eta, psi = split(q)
        else:
            u = q
    #plot(inner(u,u))
    u = project(u,VectorFunctionSpace(mesh,'CG',2))
    #plt.savefig('temp.png')
    
else:
    mesh = RectangleMesh(Point(0,0),Point(1,1),16,16)
    u = project(Constant((0,0)),VectorFunctionSpace(mesh,'CG',2))
    


if C1RUN:
    # load 9 x 9 x 4 arrays
    nn = 9
    xcase  = np.loadtxt(output_folder  + 'I1xcase.txt'  )
    ycase  = np.loadtxt(output_folder  + 'I1ycase.txt'  )
    hcase  = np.loadtxt(output_folder  + 'I1hcase.txt'  )
    d1case = np.loadtxt(output_folder  + 'I1d1case.txt' )

    xcase  = np.reshape(xcase, (nn,nn,4))
    ycase  = np.reshape(ycase, (nn,nn,4))
    hcase  = np.reshape(hcase, (nn,nn,4))
    d1case = np.reshape(d1case,(nn,nn,4))

    xx = np.zeros([nn,nn])
    yy = np.zeros([nn,nn])
    dd = np.zeros([nn,nn])
    ii = np.zeros([nn,nn])
    du = 1 + u(0.5,1)[1]

    #if du>1.5:
    # fig,ax = plt.figure(figsize=(3,4))
    fig, ax = plt.subplots(figsize=(3, 4))
    #else:
    #    plt.figure(figsize=(3,3*3/3.8))

    for i in range(nn):
        for j in range(nn):
            xtemp = xcase[i,j,1]
            ytemp = ycase[i,j,1]
            
            xy = u(xtemp,ytemp)
            xx[i,j] = xtemp + xy[0]
            yy[i,j] = ytemp + xy[1]
            dd[i,j] = np.min(d1case[i,j,:])
            ii[i,j] = np.argmin(d1case[i,j,:])

            plot_line(u,xtemp,ytemp,ii[i,j],h=0.02,LW=2)

    fmin = round(min(dd.flatten()),5)#2e-4
    fmax = round(max(dd.flatten()),5)#2.5e-3
    fsteps = 16
    c = plt.contourf(xx,yy,dd,cmap=cm.viridis,levels = np.linspace(fmin, fmax, fsteps),extend='both')
    # plt.axis('equal')
    # du = 1 + u(0.5,1)[1]

    #if du>1.5:
    plt.ylim([0,1.5])
    plt.xlim([-0.05,1.05])
    plt.yticks([0,1])
    #else:
    #    plt.yticks([0,1])
    plt.xticks([0,1])
    #plt.gca().set_yticks([0,1])
    #plt.gca().set_xticks([0,1])
    cbaxes = inset_axes(ax, width="60%", height="4%", loc='upper center') 

    #m = plt.cm.ScalarMappable(cmap=cm.viridis)
    #m.set_array(dd.flatten())
    #m.set_clim(fmin,fmax)
    # cbar = plt.colorbar(c, cax=cbaxes,boundaries=np.linspace(fmin, fmax, fsteps),ticks=[fmin,fmax],orientation='horizontal')

    

    m = plt.cm.ScalarMappable(cmap=cm.viridis)
    m.set_array(dd.flatten())
    m.set_clim(fmin,fmax)
    cbar = plt.colorbar(m, cax=cbaxes,boundaries=np.linspace(fmin, fmax, fsteps),ticks=[fmin,fmax],orientation='horizontal')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.update_ticks()
    #gcl.ax.tick_params(labelsize=12)


    #cbaxes = inset_axes(ax, width="50%", height="3%", loc='upper center')
    #cbar   = plt.colorbar(c, cax=cbaxes,orientation='horizontal')
    #cbar.formatter.set_powerlimits((0, 0))
    #cbar.formatter.set_useMathText(True)
    #cbar.update_ticks()
    #m = plt.cm.ScalarMappable(cmap=mymap)
    #m.set_array(plotfunc.vector())
    #m.set_clim(fMIN, fMAX)
    #gcl = plt.colorbar(m, cax=cbaxes,boundaries=np.linspace(fMIN, fMAX, fsteps),ticks=[fMIN,fMAX],orientation='horizontal')

    
    # gcl.ax.tick_params(labelsize=12)

    
    
    # cbar = plt.colorbar(c,label='$\mathrm{def}_1(a)$',ticks=[round(np.min(dd),2),round(np.max(dd),2)])
    #fmin = 4e-4
    #fmax = 3e-3
    #cbar = plt.colorbar(c,label='$\mathrm{def}_1(a)$',ticks=[round(np.min(dd),5),round(np.max(dd),5)])
    #cbar = plt.colorbar(c,label='$\mathrm{def}_1(a)$',ticks=[fmin,max],boundaries=np.linspace(fmin,fmax,12))
    #cbar.formatter.set_powerlimits((0, 0))
    #cbar.formatter.set_useMathText(True)
    #cbar.update_ticks()
    plt.tight_layout()
    # if du>1.5:

    #m = plt.cm.ScalarMappable(cmap=cm.jet)
    #m.set_array(dd.flatten())
    #m.set_clim(fmin, fmax)

    #cbar = plt.colorbar(m, boundaries=np.linspace(fmin, fmax, 16),ticks=[fmin,fmax])
    #cbar.formatter.set_powerlimits((0, 0))
    #cbar.formatter.set_useMathText(True)
    #cbar.update_ticks()
    #plt.yticks([0,1,2])
    #else:
    #    plt.yticks([0,1])
    #plt.xticks([0,1])
    #plt.tight_layout()
    
    #plt.xlim([-0.05,1.05])
    #plt.ylim([-0.1,2.1])
    

    plt.savefig(fname+'.pdf',dpi=300)


if C2RUN:
    # load 9 x 9 x 4 arrays
    nn = 11
    xcase  = np.loadtxt(output_folder  + 'I2xcase.txt'  )
    ycase  = np.loadtxt(output_folder  + 'I2ycase.txt'  )
    hcase  = np.loadtxt(output_folder  + 'I2hcase.txt'  )
    d1case = np.loadtxt(output_folder  + 'I2d2case.txt' )

    xcase  = np.reshape(xcase, (nn,nn,4))
    ycase  = np.reshape(ycase, (nn,nn,4))
    hcase  = np.reshape(hcase, (nn,nn,4))
    d1case = np.reshape(d1case,(nn,nn,4))

    xx = np.zeros([nn,nn])
    yy = np.zeros([nn,nn])
    dd = np.zeros([nn,nn])
    ii = np.zeros([nn,nn])

    #du = 1 + u(0.5,1)[1]
    #if du>1.5:
    #    plt.figure(figsize=(3,3.8))
    #else:
    plt.figure(figsize=(3,3*3/3.8))

    for i in range(nn):
        for j in range(nn):
            xtemp = xcase[i,j,1]
            ytemp = ycase[i,j,1]
            
            xy = u(xtemp,ytemp)
            xx[i,j] = xtemp + xy[0]
            yy[i,j] = ytemp + xy[1]
            dd[i,j] = np.min(d1case[i,j,:])
            ii[i,j] = np.argmin(d1case[i,j,:])
            itemp = ii[i,j]

            if (abs(xtemp-0.5) + abs(ytemp-0.5))>1e-4:
                plot_line(u,xtemp,ytemp,itemp,h=1/400,LW=0.5)

                if itemp == 2:
                    itemp1 = 3
                elif itemp == 3:
                    itemp1 = 2
                else:
                    itemp1 = itemp
                plot_line(u,1-xtemp,ytemp,itemp1,h=1/400,LW=0.5)
                plot_line(u,1-xtemp,1-ytemp,itemp,h=1/400,LW=0.5)
                plot_line(u,xtemp,1-ytemp,itemp1,h=1/400,LW=0.5)


    plt.plot([0.5,0.5],[0.49,0.51],'b-',linewidth=1)
    plt.contourf(xx,1-yy,dd)
    plt.contourf(1-xx,yy,dd)
    plt.contourf(1-xx,1-yy,dd)
    c = plt.contourf(xx,yy,dd)
    
    plt.axis('equal')

    plt.yticks([0.4,0.6])
    plt.xticks([0.4,0.6])
    
    # cbar = plt.colorbar(c,label='$\mathrm{def}_2(a)$',ticks=[round(np.min(dd),2),round(np.max(dd),2)])
    
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.update_ticks()
    plt.tight_layout()

    plt.savefig(fname+'CC.pdf',dpi=300)
