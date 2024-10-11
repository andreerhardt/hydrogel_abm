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

if planestrain == True:
    C1 = 1/C0
else:
    C1 = C0

# =========================================================================== #
"""Optimization options for the form compiler"""
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 5
# =========================================================================== # 
"""Boundary conditions"""
def bottom(x, on_boundary):
    tol = 1E-6
    return on_boundary and x[1] < 0.4 + tol

def top(x, on_boundary):
    tol = 1E-6
    return on_boundary and x[1] > 0.6 - tol

def mob_fun(psi):
    return mob

def mu_fun(psi):
    return 1

def extend_F(F,J):
    detF = F[0,0]*F[1,1]-F[0,1]*F[1,0]
    F33 = J / detF
    return as_tensor([[F[0,0],F[0,1],0],[F[1,0],F[1,1],0],[0,0,F33]])  
# =========================================================================== #       
"""FEA""" 
def get_energy(q):
    
    W  = q.function_space()
    
    if planestrain == True:
        if case == 3 or case == 4:
            u, p, eta, psi = split(q)
        else:
            u, p = split(q)
            psi  = phi0
    else:
        if case == 3 or case == 4:
            u, eta, psi = split(q)
        else:
            u   = q
            psi = phi0
    
    # Define elasticity stuff
    d       = u.geometric_dimension()   # dimension
    I       = Identity(d)               # Identity tensor
    I3      = Identity(3)
    F       = variable(grad(u) + I)     # Deformation gradient
    J       = det(F)                    # Determinate of F
    
    if case == 3 or case == 4:
        J_bar = (1+alpha0*(psi-phi0))
    else:
        J_bar = 1
        
    F_bar   = extend_F(F,J_bar)
    C_bar   = F_bar.T*F_bar              # (right) Cauchy-Green tensor
           
    # define free energy / lagrangian of the syste
    if case == 1 or case == 3:
        # neo-Hookean elastic energy
        e_elastic   = mu_fun(psi)/2*(tr(C_bar-I3) - 2*ln(J_bar))
    elif case == 2 or case == 4:
        # Gent elastic energy
        e_elastic   = -mu_fun(psi)/2*(Jm*ln(1-tr(C_bar-I3)/Jm) + 2*ln(J_bar))        
    if case == 3 or case == 4:
        # hydrogel chemical energy
        e_phase     = (C1/2*(alpha0*psi/(J_bar*Nmono)*ln(alpha0*psi/J_bar) + (1-alpha0*psi/J_bar)*ln(1-alpha0*psi/J_bar) - ln(1-alpha0*phi0)) + eps/2*inner(grad(psi),grad(psi)))*J_bar
        e_free      = (e_elastic + e_phase)
    else:
        e_free      = e_elastic
        
    if planestrain == True:
        e_free += p*(J-J_bar)
        
    P = diff(e_free,F)

    E_free  = e_free*dx
    Efree   = assemble(E_free)
    
    facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, mesh.domains())
    facets.set_all(0)
    AutoSubDomain(top).mark(facets, 1)
    AutoSubDomain(bottom).mark(facets, 2)
    ds = Measure("ds", subdomain_data=facets)
    
    ny          = Expression(("0.0","1.0"), degree = 2)
    f_int       = assemble(inner(dot(P,ny),ny)*ds(1))
    
    return Efree,f_int

energies=[]

x1 = 0.49
x0 = 0.51
d0 = (0.001*0.001)
f0 = traction / (2*pi*d0)


f1 = File(output_folder + "disp.pvd")
f2 = File(output_folder + "conc.pvd")
f3 = File(output_folder + "strain.pvd")
f4 = File(output_folder + "force.pvd")
f5 = File(output_folder + "base.pvd")

for i in range(0,n_steps+1):  
    if i in range(0,n_steps+1):

        
        mesh,q,W = load_state(output_folder + 'state'+str(i),case,planestrain)
        E_free,f_int = get_energy(q)
        t = (i-1)*dt
        energies.append([i,t,E_free,f_int])

        print("iteration ",i," of ",n_steps," f_int:",f_int)

        if True:
            f0 = (i/n_steps) * traction / (2*pi*d0)
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

            basemesh = RectangleMesh(Point(0.4,0.4),Point(0.6,0.6),20,20,"left/right")
            disp = project(u,VectorFunctionSpace(basemesh,'CG',1))
            ALE.move(basemesh,disp)

            o1  = project(u,Vv)
            o2  = project(psi,Vs)
            o3  = project(Identity(2)+grad(u),Vt)
            o4  = project(force,Vv)

            ALE.move(mesh,o1)

            o1.rename("u","u")
            o2.rename("c","c")
            o3.rename("F","F")
            o4.rename("f","f")
            disp.rename("disp","disp")

            f1 << (o1,t)
            f2 << (o2,t)
            f3 << (o3,t)
            f4 << (o4,t)
            f5 << (disp,t)
        
E = list(zip(*energies))
np.save(output_folder + "energies",E)