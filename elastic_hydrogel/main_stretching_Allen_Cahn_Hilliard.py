#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:48:46 2023

@authors:    André H. Erhardt (andre.erhardt@wias-berlin.de, https://orcid.org/0000-0003-4389-8554)
             Dirk Peschka (dirk.peschka@wias-berlin.de, https://orcid.org/0000-0002-3047-1140)

"""

import numpy as np
import pylab
import sys
from fenics import *
from abm_module import *
from os import mkdir

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 5

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

try:
    mkdir(output_folder)
except:
    print("Folder already exists")

write_dictionary(output_folder + 'run_pars.json',abm_pars)

# =========================================================================== #
# main functions for the solver

# boundary marker
def bottom(x, on_boundary):
    tol = 1E-6
    return on_boundary and x[1] < tol
def top(x, on_boundary):
    tol = 1E-6
    return on_boundary and x[1] > 1 - tol

# time-dependent Dirichlet boundary condition
def move_boundary(t):
    if (t<t_final):
        scale = alpha*(t/t_final)
    else:
        scale = alpha
    return Expression(("0","scale"), scale=scale, degree=2)

# Cahn-Hilliard mobility
def mob_fun(psi):
    return mob

# Concentration-dependent bulk modulus
def mu_fun(psi):
    return 1

# extension of 2x2 deformation gradient to a 3x3 deformation gradient for
# thin or thick sheets (plane strain vs plain stress approximation)
def extend_F(F,J):
    detF = F[0,0]*F[1,1]-F[0,1]*F[1,0]
    F33 = J/detF
    return as_tensor([[F[0,0],F[0,1],0],[F[1,0],F[1,1],0],[0,0,F33]])

# single time-step for the nonlinearly coupled problem coupling 
# hydrogel and nonlinear elasticity 
def evolve(old_q,t,dt):
    # define functions and test functions and components
    q,dq  = Function(W),TestFunction(W)
    if planestrain == True:
        if case == 3 or case == 4:
            u, p, eta, psi                   = split(q)
            old_u, old_p, old_eta, old_psi   = split(old_q)
            du, dp, deta, dpsi               = split(dq)
        else:
            u, p            = split(q)
            old_u, old_p    = split(old_q)
            du , dp         = split(dq)
            psi   = phi0
    else:
        if case == 3 or case == 4:
            u, eta, psi = split(q)
            old_u, old_eta, old_psi = split(old_q)
            du, deta, dpsi = split(dq)
        else:
            u     = q
            old_u = old_q
            du    = dq
            psi   = phi0
    
    # define elasticity stuff
    d       = u.geometric_dimension()   # dimension
    I       = Identity(d)               # Identity tensor
    I3      = Identity(3)
    F       = grad(u) + I               # Deformation gradient
    J       = det(F)                    # Determinate of F
    
    if case == 3 or case == 4:
        J_bar = (1.0+alpha0*(psi-phi0))
    else:
        J_bar = 1.0

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
        k0          = ln(alpha0*phi0)
        e_phase     = (C0/2*(alpha0*psi/(J_bar*Nmono)*ln(alpha0*psi/J_bar) + (1-alpha0*psi/J_bar)*ln(1-alpha0*psi/J_bar) - k0) + eps/2*inner(grad(psi),grad(psi)))*J_bar

        e_free      = (e_elastic + e_phase)
    else:
        e_free      = e_elastic
        
    if planestrain == True:
        # Lagrange multiplier for constraint
        e_free += p*(J-J_bar)

    E_free = e_free*dx
    
    # compute derivative of free energy / lagrangian
    Res = derivative(E_free, q, dq) 
        
    # add mixed terms for time-dependent formulation
    if case == 3 or case == 4:
        dot_psi = (psi-old_psi)/dt  
        Res += -inner(eta,dpsi)*dx
        Res += inner(dot_psi, deta)*dx + mob_fun(old_psi)*inner(grad(eta),grad(deta))*dx 
        if AllenCahn == True:
            Res += mob*eta*deta*dx
    
    # add boundary conditions
    if planestrain  == True or case == 3 or case == 4:
        bc_U = DirichletBC(W.sub(0), move_boundary(t), top)
        bc_D = DirichletBC(W.sub(0), Constant((0.0,0.0)), bottom)
    else:
        bc_U = DirichletBC(W, move_boundary(t), top)
        bc_D = DirichletBC(W, Constant((0.0,0.0)), bottom)

    # solve    
    Jac = derivative(Res, q) 

    q.assign(old_q)
    bc          = [bc_U, bc_D]
    NEWTON_tol  = 1e-6
    solve(Res==0, q, bc, J=Jac, solver_parameters={"newton_solver":{"absolute_tolerance": NEWTON_tol, "maximum_iterations": 10}})
    E_free      = assemble(E_free)
    return q, E_free

# =========================================================================== # 
# main loop

# create mesh, function space, initial data
if planestrain  == True:
    if case == 3 or case == 4:
        initial     = Expression(("0","0","0","0","phi0"),phi0=phi0, degree=2)
        correction  = Expression(("0","alpha*dt*x[1]/t_final","0","0","0"), dt=dt,alpha=alpha,t_final=t_final, degree=2)
    else:
        initial     = Expression(("0","0","0"), degree=2)
        correction  = Expression(("0","alpha*dt*x[1]/t_final","0"), dt=dt,alpha=alpha,t_final=t_final, degree=2)
else:
    if case == 3 or case == 4:
        initial     = Expression(("0","0","0","phi0"),phi0=phi0, degree=2)
        correction  = Expression(("0","alpha*dt*x[1]/t_final","0","0"), dt=dt,alpha=alpha,t_final=t_final, degree=2)
    else:
        initial     = Expression(("0","0"), degree=2)
        correction  = Expression(("0","alpha*dt*x[1]/t_final"), dt=dt,alpha=alpha,t_final=t_final, degree=2)

t        = 0.0
mesh     = RectangleMesh(Point(0,0),Point(1,1),100,100,"left/right")
W        = get_space(mesh,case,planestrain)
upd      = project(correction,W)
old_q    = interpolate(initial, W)
q        = interpolate(initial, W)

# runing the main time loop by stretching the hydrogel/elastic solid
for i in range(1,n_steps+1):  
    print("iteration ",i," of ",n_steps)
    # call solve
    q, E_free = evolve(old_q,t,dt)
    old_q.assign(q)
   
    # output
    if i in range(1,n_steps+1,5):
        save_state(mesh,q,output_folder + 'state'+str(i),t)

    if (t<t_final):
        old_q.vector().set_local(old_q.vector() + upd.vector())
    t += dt
