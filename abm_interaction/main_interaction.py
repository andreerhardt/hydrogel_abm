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

random.seed(30)
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
NEWTON_SOLVER = abm_pars["NEWTON_SOLVER"]

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
"""Boundary conditions"""
def bottom(x, on_boundary):
    tol = 1E-6
    return on_boundary and x[1] < tol
def top(x, on_boundary):
    tol = 1E-6
    return on_boundary and x[1] > 1 - tol

# =========================================================================== #
class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs, node1_x,node1_y,node2_x,node2_y):
        NonlinearProblem.__init__(self)
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        self.node1_x = node1_x
        self.node1_y = node1_y
        self.node2_x = node2_x
        self.node2_y = node2_y

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        if planestrain  == True or case == 3 or case == 4:
            point_load_1_x = PointSource(W.sub(0).sub(0), self.node1_x) #"""node1 point sources for x-component"""
            point_load_1_y = PointSource(W.sub(0).sub(1), self.node1_y) #"""node1 point sources for y-component"""
            point_load_2_x = PointSource(W.sub(0).sub(0), self.node2_x) #"""node2 point sources for x-component"""
            point_load_2_y = PointSource(W.sub(0).sub(1), self.node2_y) #"""node2 point sources for y-component"""
        else:    
            point_load_1_x = PointSource(W.sub(0), node1_x) #"""node1 point sources for x-component"""
            point_load_1_y = PointSource(W.sub(1), node1_y) #"""node1 point sources for y-component"""
            point_load_2_x = PointSource(W.sub(0), node2_x) #"""node2 point sources for x-component"""
            point_load_2_y = PointSource(W.sub(1), node2_y) #"""node2 point sources for y-component"""
        
        point_load = [point_load_1_x,point_load_1_y,point_load_2_x,point_load_2_y]
        for load in point_load:
            load.apply(b)
        for bc in self.bcs:
            bc.apply(b,x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A,x)   
# =========================================================================== #            
class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, mesh.mpi_comm(),
                              PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        PETScOptions.set("ksp_type", "preonly")
        PETScOptions.set("pc_type", "lu")
        PETScOptions.set("pc_factor_mat_solver_type", "mumps")

        self.linear_solver().set_from_options()

    def update_solution(self, x, dx, relaxation_parameter, nonlinear_problem, iteration):
        tau = 1.0
        theta = min(sqrt(2.0*tau/norm(dx, norm_type="l2", mesh=W.mesh())), 1.0)
        info("Newton damping parameter: %.3e" % theta)
        x.axpy(-theta, dx)
# =========================================================================== #   
def move_boundary(t):
    if (t<t_final):
        scale = round(alpha*t,3)
    else:
        scale = round(alpha*t_final,3)
    return Expression(("0","scale"), scale=scale, degree=2)

def mob_fun(psi):
    return mob

# Concentration-dependent bulk modulus
def mu_fun(psi):
    return 1

def extend_F(F,J):
    detF = F[0,0]*F[1,1]-F[0,1]*F[1,0]
    F33 = J / detF
    return as_tensor([[F[0,0],F[0,1],0],[F[1,0],F[1,1],0],[0,0,F33]])
# =========================================================================== # 
"""FEA""" 
def evolve(old_q,t,dt,node1_x=[],node1_y=[],node2_x=[],node2_y=[],CELLS=True):
     
    W = old_q.function_space()
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
        e_phase     = (C1/2*(alpha0*psi/(J_bar*Nmono)*ln(alpha0*psi/J_bar) + (1-alpha0*psi/J_bar)*ln(1-alpha0*psi/J_bar) - k0) + eps/2*inner(grad(psi),grad(psi)))*J_bar

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
    if CELLS:   
        problem = Problem(Jac, Res, bc,node1_x,node1_y,node2_x,node2_y)
        solver = NewtonSolver()
        solver.parameters['linear_solver'] = NEWTON_SOLVER
        # solver = CustomSolver()
        solver.parameters['relative_tolerance'] = 1e-6
        solver.parameters['absolute_tolerance'] = 1e-6
        solver.solve(problem, q.vector())
    else:
        NEWTON_tol  = 1e-6
        solve(Res==0, q, bc, J=Jac, solver_parameters={"newton_solver":{"absolute_tolerance": NEWTON_tol, "maximum_iterations": 10}})
    

    
    E_free      = assemble(E_free)
    return q, E_free
# =========================================================================== # 

# =========================================================================== #
"""defining the agent-based problem""" 
t        = 0
mesh     = RectangleMesh(Point(0,0),Point(1,1),100,100,"left/right")
W        = get_space(mesh,case,planestrain)

"""Optimization options for the form compiler"""
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 5

"""initial values"""
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

"""initialisation of the agent-based model"""
random.seed(2)  
abm         = ABM(width, height,empty_ratio,traction)             
W           = get_space(mesh,case,planestrain)
q           = interpolate(initial, W)
old_q       = interpolate(initial, W)
upd         = project(correction,W)

orient_name = ["vert  ","horiz ","diag 1","diag 2"]

output_folder = abm_pars["output_folder"]

if C0RUN:
    for i in range(1,n_steps+1):  
        print("iteration ",i," of ",n_steps)
        # call solve
        q, E_free = evolve(old_q,t,dt,CELLS=False)
        old_q.assign(q)
   
        # output
        if (t<t_final):
            old_q.vector().set_local(old_q.vector() + upd.vector())
        t += dt
    
    save_state(mesh,q,output_folder + 'deformed_C0',t)

if C1RUN:
    print("C1RUN start")
    xcase  = []
    ycase  = []
    hcase  = []
    d1case = []
    d2case = []
    for xx in np.linspace(0.02,0.98,9):
        for yy in np.linspace(0.02,0.98,9):
            for h in [1,2,3,4]:
                n_dipole = 1
                xc = [round(xx,2)]
                yc = [round(yy,2)]
                hc = [h]   
        
                (n1list,n2list,clist,node1_x,node1_y,node2_x,node2_y,ndipole) = abm.set_dipole_direction_point_source(n_dipole,xc,yc,hc)
                q, E_free  = evolve(q,t,dt,node1_x,node1_y,node2_x,node2_y)
                
                if (case==3) or (case==4):
                    q, E_free  = evolve(q,t,dt,node1_x,node1_y,node2_x,node2_y)
                    # q, E_free  = evolve(q,t,dt,node1_x,node1_y,node2_x,node2_y)
                    
                deform_new = abm.manual_deformation(n_dipole,xc,yc,hc,q)
                xcase.append(xc[0])
                ycase.append(yc[0])
                hcase.append(h)
                d1case.append(deform_new[0])
                print("case:",case,"x:",xc[0]," y:",yc[0]," h:",orient_name[h-1]," d1:",deform_new[0])    

    np.savetxt(output_folder  + 'I1xcase.txt', xcase, delimiter=' ') 
    np.savetxt(output_folder  + 'I1ycase.txt', ycase, delimiter=' ') 
    np.savetxt(output_folder  + 'I1hcase.txt', hcase, delimiter=' ') 
    np.savetxt(output_folder  + 'I1d1case.txt', d1case, delimiter=' ')

if C2RUN:
    print("C2RUN start")
    xcase = []
    ycase = []
    hcase = []
    d1case = []
    d2case = []
    for xx in np.linspace(0.4,0.5,11):
        for yy in np.linspace(0.4,0.5,11):
            for h in [1,2,3,4]:
                n_dipole = 2
                xc = [round(xx,2),0.5]
                yc = [round(yy,2),0.5]
                hc = [h,1]   
                (n1list,n2list,clist,node1_x,node1_y,node2_x,node2_y,ndipole) = abm.set_dipole_direction_point_source(n_dipole,xc,yc,hc)
                q, E_free  = evolve(q,t,dt,node1_x,node1_y,node2_x,node2_y)
                if (case==3) or (case==4):
                    q, E_free  = evolve(q,t,dt,node1_x,node1_y,node2_x,node2_y)
                    # q, E_free  = evolve(q,t,dt,node1_x,node1_y,node2_x,node2_y)
                    
                deform_new = abm.manual_deformation(n_dipole,xc,yc,hc,q)
                xcase.append(xc[0])
                ycase.append(yc[0])
                hcase.append(h)
                d1case.append(deform_new[0])
                d2case.append(deform_new[1])
                print("case:",case,"x:",xc[0]," y:",yc[0]," h:",orient_name[h-1]," d1:",deform_new[0]," d2:",deform_new[1])      
    np.savetxt(output_folder  + 'I2xcase.txt', xcase, delimiter=' ') 
    np.savetxt(output_folder  + 'I2ycase.txt', ycase, delimiter=' ') 
    np.savetxt(output_folder  + 'I2hcase.txt', hcase, delimiter=' ') 
    np.savetxt(output_folder  + 'I2d1case.txt', d1case, delimiter=' ') 
    np.savetxt(output_folder  + 'I2d2case.txt', d2case, delimiter=' ') 