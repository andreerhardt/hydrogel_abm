#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:20:02 2023

@authors:    André H. Erhardt (andre.erhardt@wias-berlin.de, https://orcid.org/0000-0003-4389-8554)

"""

import numpy as np
import pylab
import sys
from fenics import *
from abm_module import *
from os import mkdir
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['font.size'] = '16'
plt.rcParams["font.family"] = "Times New Roman"

if len(sys.argv) == 1:
    print("Reading default.json")
    abm_pars = read_dictionary('default.json')
else:
    in_json = sys.argv[1]
    print("Reading ",in_json)
    abm_pars = read_dictionary(in_json)

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
name          = abm_pars["name"]

if planestrain == True:
    C1 = 1/C0
else:
    C1 = C0

x1 = 0.49
x0 = 0.51
d0 = (0.001*0.001)
f0 = traction / (2*pi*d0)


mesh,q,W = load_state(output_folder + 'state'+str(n_steps),case,planestrain)
force = Expression(("f0*(exp(-(pow(x[0]-x0,2)+pow(x[1]-0.5,2))/(2*d0))-exp(-(pow(x[0]-x1,2)+pow(x[1]-0.5,2))/(2*d0)))","0"),degree=2,d0=d0,x0=x0,x1=x1,f0=f0)

Vv = VectorFunctionSpace(mesh,'CG',2)
Vs = FunctionSpace(mesh,"CG",2)
Vt = TensorFunctionSpace(mesh,"CG",1,shape=(2,2))

if planestrain == True:
    if case==3 or case == 4:
        u, p, eta, psi = split(q)
    else:
        u, p = split(q)
        psi  = phi0
else:
    if case==3 or case == 4:
        u, eta, psi = split(q)
    else:
        u   = q
        psi = phi0


psi = project(psi,FunctionSpace(mesh,'CG',2))
u   = project(u,VectorFunctionSpace(mesh,'CG',2))

basemesh = RectangleMesh(Point(0.4,0.4),Point(0.6,0.6),100,100,"left/right")
disp = project(u,VectorFunctionSpace(basemesh,'CG',1))

dx1 = u(0.49,0.5)
dx2 = u(0.51,0.5)

dx1 = dx1[0]
dx2 = dx2[0]

ALE.move(mesh,u)
ALE.move(basemesh,disp)

o1  = project(u,Vv)
o2  = project(psi,Vs)
o3  = project(Identity(2)+grad(u),Vt)
o4  = project(force,Vv)
fmin = 0.0
fmax = 0.4
fsteps = 64

ax = plt.subplot(1,1,1)

c = plot(psi,cmap=cm.bwr_r,vmin=fmin,vmax=fmax,levels = np.linspace(fmin,fmax, fsteps))
plot(mesh,linewidth=0.1,zorder=2)
plot(disp,scale=.15,cmap=cm.jet,zorder=3)
        
#plt.colorbar(c,ticks=np.linspace(fmin, fmax, 3),label='$\hat{c}$')
plt.xlim([0.47,0.53])
plt.ylim([0.47,0.53])

plt.gca().set_yticks([])
plt.gca().set_xticks([])



plt.plot([0.49+dx1,0.51+dx2],[0.5,0.5],'r-',lw=2,path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
plt.plot([0.49+dx1,0.51+dx2],[0.5,0.5],'ro',ms=6,path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])

plt.plot([0.5],[0.5],'bo',lw=1)

cbaxes = inset_axes(ax, width="80%", height="6%", loc='upper center') 
m = plt.cm.ScalarMappable(cmap=cm.bwr_r)
m.set_array(psi.vector())
m.set_clim(fmin, fmax)
gcl = plt.colorbar(m, cax=cbaxes,boundaries=np.linspace(fmin,fmax, fsteps),ticks=[fmin,0.2,fmax],orientation='horizontal',label='$\hat{c}$')

plt.savefig(name + '.pdf',dpi=600,bbox_inches='tight', pad_inches=0.05)
plt.savefig(name + '.jpeg',dpi=230,bbox_inches='tight', pad_inches=0.05)