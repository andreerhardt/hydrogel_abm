from fenics import *
import pickle
import itertools
import random 
from random import randint
import numpy as np
import copy
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors

# =========================================================================== #
""""Define the function space"""
def get_space(mesh,case=1,planestrain=True):
    P2 = VectorElement("P", mesh.ufl_cell(), 2)
    P1 = FiniteElement("P", mesh.ufl_cell(), 1)
    if planestrain == True:
        if case == 3 or case == 4:
            TH = MixedElement([P2,P1,P1,P1]) 
        else:
            TH = MixedElement([P2,P1]) 
    else:
        if case == 3 or case == 4:
            TH = MixedElement([P2,P1,P1]) 
        else:
            TH = P2 
                    
    W  = FunctionSpace(mesh, TH)
    return W

def save_state(mesh,q,fname,t=0):
    ff=HDF5File(mesh.mpi_comm(),fname  + '.h5', 'w')
    ff.write(mesh,"mesh")   
    ff.write(q,"q",t)
    ff.close()
    
def load_state(fname,case=1,planestrain=True):
    mesh = Mesh()
    f=HDF5File(mesh.mpi_comm(),fname + '.h5', 'r')
    f.read(mesh,"mesh",False)   
    W    = get_space(mesh,case,planestrain)
    q    = Function(W)
    f.read(q,"q")
    return mesh,q,W

def read_dictionary(fname):
    with open(fname) as json_file:
        pars = json.load(json_file)   
    return pars

def write_dictionary(fname,mydict):
    with open(fname, 'w') as fp:
        json.dump(mydict, fp,sort_keys=True, indent=4)

"""Output routine"""   
def output(q, t = 0, case = 1):
    
    Vu   = VectorFunctionSpace(mesh,'CG',2)
    Vpsi = FunctionSpace(mesh,'CG',1)
    
    if planestrain == True:
        if case==3 or case == 4:
            u, p, eta, psi  = split(q)
            outpsi  = project(psi,Vpsi)
            outpsi.rename("psi","psi")
        else:
            u, p            = split(q)
    else:
        if case==3 or case == 4:
            u, eta, psi     = split(q)
            outpsi  = project(psi,Vpsi)
            outpsi.rename("psi","psi")
        else:
            u     = q
        
    outu   = project(u,Vu)
    outu.rename("disp","disp")
            
    outF = project(grad(outu),TensorFunctionSpace(mesh, "CG", 1, shape=(2,2)))
    outF.rename("F","F")
    
    if planestrain == True or case == 3 or case == 4:
        ALE.move(mesh, q.sub(0))
    else:
        ALE.move(mesh, q)

    file1 << (outu,t)
    if case==3 or case==4:
        file2 << (outpsi,t)
    file3 << (outF,t)
    
def compute_dipoles(mesh,node1_x,node1_y,node2_x,node2_y,traction):
    X = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(X)
    v = TestFunction(X)
    
    a = inner(u,v)*dx
    L = inner(Constant((0,0)),v)*dx
    A, b = assemble_system(a, L)
    
    point_load_1_x = PointSource(X.sub(0), node1_x) 
    point_load_1_y = PointSource(X.sub(1), node1_y) 
    point_load_2_x = PointSource(X.sub(0), node2_x) 
    point_load_2_y = PointSource(X.sub(1), node2_y) 
    point_load = [point_load_1_x,point_load_1_y,point_load_2_x,point_load_2_y]
    for load in point_load:
        load.apply(b)
  
    nnn = 6.661192481723062
    u = Function(X)
    solve(A, u.vector(), b)
    dipoles = assemble((abs(u.sub(0))+abs(u.sub(1)))*dx)
    dp      = dipoles /(abs(traction)*nnn)
    return dipoles,dp

def plot_magnitude_cell_without_grid(title,output_folder,file_name,n,n1list,n2list,cells,q,mesh):
    nodes = [n1list,n2list]
    u    = q.sub(0)
    Vu   = FunctionSpace(mesh,'P',2)
    Vpsi = FunctionSpace(mesh,'P',1)
    
    u_magnitude = sqrt(inner(u,u))
    u_magnitude = project(u_magnitude,Vu)
    maximum = u_magnitude.vector().max()
    minimum = u_magnitude.vector().min()
    
    fig, ax = plt.subplots(figsize=(8.5,7))
    plt.rcParams['font.size'] = '20'
    surf = plot(u_magnitude, cmap='gist_rainbow',vmin = 0, vmax = maximum) 
    surf.set_clim(0,maximum)
     
    for j in range(np.size(nodes,1)):
        plt.arrow(nodes[0][j][0], nodes[0][j][1],cells[j][0]-nodes[0][j][0],cells[j][1]-nodes[0][j][1],linestyle = '-',linewidth = 0.25,color = 'k',shape='right',head_width=0.004,head_length=0.002)
        plt.arrow(nodes[1][j][0], nodes[1][j][1],cells[j][0]-nodes[1][j][0],cells[j][1]-nodes[1][j][1],linestyle = '-',linewidth = 0.25,color = 'k',shape='right',head_width=0.004,head_length=0.002)
        ax.scatter(nodes[0][j][0], nodes[0][j][1], color='k',s=0.25)#'r'
        ax.scatter(nodes[1][j][0], nodes[1][j][1], color='k',s=0.25)#'b'
    for agent in cells:
        ax.scatter(agent[0], agent[1], color='k',s=1)
        
    ax.set_ylabel('y coordinate', fontsize=20) # Y label
    ax.set_xlabel('x coordinate', fontsize=20) # X 
    ax.tick_params(labelsize=20)
    plt.setp(ax, xticks=[-0.05,0,0.25,0.5,0.75,1,1.05], xticklabels=['',0,0.25,0.5,0.75,1,''])
    plt.setp(ax, yticks=[-0.05,0,0.25,0.5,0.75,1,1.05], yticklabels=['',0,0.25,0.5,0.75,1,''])
    
    cbar = fig.colorbar(surf, ticks = [0,maximum])
    cbar.ax.set_yticklabels(['0', format(maximum,'.1e')],fontsize = 10) 
    cbar.ax.set_ylabel('magnitude of the displacement $u$', loc='center')
    plt.savefig(output_folder + file_name + '_final_configuration'+".pdf") 
    
def plot_psi(title,output_folder,file_name,n,n1list,n2list,cells,q,mesh,planestrain):
    nodes = [n1list,n2list]
    if planestrain == True:
        psi    = q.sub(3)
    else:
        psi    = q.sub(2)
    Vpsi = FunctionSpace(mesh,'P',1)
    
    u_magnitude = project(psi,Vpsi)
    maximum = u_magnitude.vector().max()
    minimum = u_magnitude.vector().min()
    
    
    fig, ax = plt.subplots(figsize=(8.5,7))
    plt.rcParams['font.size'] = '20'
    surf = plot(u_magnitude, cmap='Blues',vmin = minimum, vmax = maximum) 
    surf.set_clim(minimum,maximum)
     
    for j in range(np.size(nodes,1)):
        plt.arrow(nodes[0][j][0], nodes[0][j][1],cells[j][0]-nodes[0][j][0],cells[j][1]-nodes[0][j][1],linestyle = '-',linewidth = 0.25,color = 'k',shape='right',head_width=0.004,head_length=0.002)
        plt.arrow(nodes[1][j][0], nodes[1][j][1],cells[j][0]-nodes[1][j][0],cells[j][1]-nodes[1][j][1],linestyle = '-',linewidth = 0.25,color = 'k',shape='right',head_width=0.004,head_length=0.002)
        ax.scatter(nodes[0][j][0], nodes[0][j][1], color='k',s=0.25)
        ax.scatter(nodes[1][j][0], nodes[1][j][1], color='k',s=0.25)
    for agent in cells:
        ax.scatter(agent[0], agent[1], color='k',s=1)
        
    ax.set_ylabel('y coordinate', fontsize=20) 
    ax.set_xlabel('x coordinate', fontsize=20) 
    ax.tick_params(labelsize=20)
    plt.setp(ax, xticks=[-0.05,0,0.25,0.5,0.75,1,1.05], xticklabels=['',0,0.25,0.5,0.75,1,''])
    plt.setp(ax, yticks=[-0.05,0,0.25,0.5,0.75,1,1.05], yticklabels=['',0,0.25,0.5,0.75,1,''])
    
    cbar = fig.colorbar(surf, ticks = [minimum,maximum])
    #cbar.ax.set_yticklabels(['0', format(maximum,'.1e')],fontsize = 10) 
    cbar.ax.set_ylabel('concentration $c$', loc='center')
    plt.savefig(output_folder + file_name + '_final_configuration_solvent'+".pdf") 
    
def plot_gradu(title,output_folder,file_name,n,n1list,n2list,cells,q,mesh,case,planestrain):
    nodes          = [n1list,n2list]
    if planestrain == True or case == 3 or case == 4:
        F          = grad(q.sub(0)) 
    else:
        F          = grad(q)
    deform_grad_xx = project(F[0,0],FunctionSpace(mesh, "Lagrange", 1))
    deform_grad_yy = project(F[1,1],FunctionSpace(mesh, "Lagrange", 1))
    
    maximum = max(deform_grad_xx.vector().max(),deform_grad_yy.vector().max())
    minimum = min(deform_grad_xx.vector().min(),deform_grad_yy.vector().min())
    
    cmap = plt.get_cmap('jet')
    vmin = min(minimum,-maximum)  #minimum value to show on colobar
    vmax = max(-minimum,maximum)  #maximum value to show on colobar
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax =vmax)
    #generate colors from original colormap in the range equivalent to [vmin, vamx] 
    colors = cmap(np.linspace(1.-(vmax-vmin)/float(vmax), 1, cmap.N))
    # Create a new colormap from those colors
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_jet', colors)
    
    fig, ax = plt.subplots(figsize=(23.5,8.5))
    sub1 = plt.subplot(121)
    surf = plot(deform_grad_xx,cmap=cmap)
    sub1.set_ylabel('y coordinate', fontsize=20)
    sub1.set_xlabel('x coordinate', fontsize=20)
    sub1.tick_params(labelsize=20)
    plt.setp(sub1, xticks=[-0.05,0,0.25,0.5,0.75,1,1.05], xticklabels=['',0,0.25,0.5,0.75,1,''])
    plt.setp(sub1, yticks=[-0.05,0,0.25,0.5,0.75,1,1.05], yticklabels=['',0,0.25,0.5,0.75,1,''])
    surf.set_clim(minimum,maximum)
    for j in range(np.size(nodes,1)):
        plt.arrow(nodes[0][j][0], nodes[0][j][1],cells[j][0]-nodes[0][j][0],cells[j][1]-nodes[0][j][1],linestyle = '-',linewidth = 0.25,color = 'k',shape='right',head_width=0.004,head_length=0.002)
        plt.arrow(nodes[1][j][0], nodes[1][j][1],cells[j][0]-nodes[1][j][0],cells[j][1]-nodes[1][j][1],linestyle = '-',linewidth = 0.25,color = 'k',shape='right',head_width=0.004,head_length=0.002)
        sub1.scatter(nodes[0][j][0], nodes[0][j][1], color='k',s=0.25)#'r'
        sub1.scatter(nodes[1][j][0], nodes[1][j][1], color='k',s=0.25)#'b'
    for agent in cells:
        sub1.scatter(agent[0], agent[1], color='k',s=1)
    sub1.set_title('magnitude of $\\nabla u_{11}$', fontsize=15)
    
    # create some axes to put the colorbar to
    cax, _  = mpl.colorbar.make_axes(plt.gca())
    cbar    = mpl.colorbar.ColorbarBase(cax, cmap=color_map, norm=norm,)
    cbar.set_ticks([minimum,maximum])
    cbar.set_ticklabels([minimum,maximum])
    cbar.ax.set_yticklabels([format(minimum,'.1e'), format(maximum,'.1e')],fontsize = 10) 
    
    sub2 = plt.subplot(122)
    surf = plot(deform_grad_yy,cmap=cmap)
    sub2.set_ylabel('y coordinate', fontsize=20) 
    sub2.set_xlabel('x coordinate', fontsize=20) 
    sub2.tick_params(labelsize=20)
    plt.setp(sub2, xticks=[-0.05,0,0.25,0.5,0.75,1,1.05], xticklabels=['',0,0.25,0.5,0.75,1,''])
    plt.setp(sub2, yticks=[-0.05,0,0.25,0.5,0.75,1,1.05], yticklabels=['',0,0.25,0.5,0.75,1,''])
    surf.set_clim(minimum,maximum) 
    for j in range(np.size(nodes,1)):
        plt.arrow(nodes[0][j][0], nodes[0][j][1],cells[j][0]-nodes[0][j][0],cells[j][1]-nodes[0][j][1],linestyle = '-',linewidth = 0.25,color = 'k',shape='right',head_width=0.004,head_length=0.002)
        plt.arrow(nodes[1][j][0], nodes[1][j][1],cells[j][0]-nodes[1][j][0],cells[j][1]-nodes[1][j][1],linestyle = '-',linewidth = 0.25,color = 'k',shape='right',head_width=0.004,head_length=0.002)
        sub2.scatter(nodes[0][j][0], nodes[0][j][1], color='k',s=0.25)#'r'
        sub2.scatter(nodes[1][j][0], nodes[1][j][1], color='k',s=0.25)#'b'
    for agent in cells:
        sub2.scatter(agent[0], agent[1], color='k',s=1)
    sub2.set_title('magnitude of $\\nabla u_{22}$', fontsize=15)
    
    # create some axes to put the colorbar to
    cax, _  = mpl.colorbar.make_axes(plt.gca())
    cbar    = mpl.colorbar.ColorbarBase(cax, cmap=color_map, norm=norm,)
    cbar.set_ticks([minimum, maximum])
    cbar.set_ticklabels([minimum, maximum])
    cbar.ax.set_yticklabels([format(minimum,'.1e'), format(maximum,'.1e')],fontsize = 10) 
    
    plt.savefig(output_folder + file_name + '_final_configuration_gradu'+".pdf")

# =========================================================================== #
"""Agent-based model"""
class ABM:
    def __init__(self,width,height,empty_ratio,traction,species = 1):
        self.width = width # float
        self.height = height  # float
        self.empty_ratio = empty_ratio # float
        self.species = species # integer
        self.empty_nodes = []  # list
        self.agents = {}       # set
        self.old_agents = {}   # set    
        self.traction   = traction
        
    def save_state(self,fname):
        data = {"width":self.width,
                "height":self.height,
                "empty_ratio":self.empty_ratio,
                "species":self.species,
                "empty_nodes":self.empty_nodes,
                "agents":self.agents,
                "traction":self.traction,
                "old_agents":self.old_agents}
        ffile = open(fname + '.abm', 'wb')
        pickle.dump(data, ffile)
        ffile.close()
        
    def load_state(self,fname):
        ffile = open(fname + '.abm', 'rb')
        data  = pickle.load(ffile)
        self.width       = data["width"]
        self.height      = data["height"]
        self.empty_ratio = data["empty_ratio"]
        self.species     = data["species"]
        self.empty_nodes = data["empty_nodes"]
        self.agents      = data["agents"]
        self.traction    = data["traction"]
        self.old_agents  = data["old_agents"]
        
    def return_cells(self):
        cells = []
        for agent in self.agents:
            cells.append(agent)
        return cells
    
    def align_cells(self):
        for agent in self.agents:
            x = agent[0]
            y = agent[1]
            tol = 1e-4
            if abs(x) < tol or abs(x-1) < tol: #"""vertical""" 
                self.agents[agent] = 1 
            elif abs(y) < tol or abs(y-1) < tol: #"""horizontal""" 
                self.agents[agent] = 2

    def shuffle_orientation(self):
        for agent in self.agents:
            x = agent[0]
            y = agent[1]
            h = randint(1, 4) #4
            self.agents[agent] = h   
    
    """seeding of cells"""
    def seeding(self):
        all_nodes = list(itertools.product([x/100 for x in range(0, self.width+1)],[x/100 for x in range(0, self.height+1)], repeat=1))
        # remove corners
        ilist = [(0,0),(1,0),(0,1),(1,1)]
        for i in ilist:
            index = all_nodes.index(i)
            del all_nodes[index]
        random.shuffle(all_nodes)
        
        n_empty = int( self.empty_ratio * len(all_nodes) )
        self.empty_nodes = all_nodes[:n_empty]
        remaining_nodes = all_nodes[n_empty:]
        nodes_by_specie = [remaining_nodes[i::self.species] for i in range(self.species)]
        for i in range(self.species):
            self.agents.update(dict(zip(nodes_by_specie[i], [i+1]*len(nodes_by_specie[i]))))
            
        self.shuffle_orientation()
        self.align_cells()
    
    """determine empty neighbours"""
    def neighbour_empty(self, x, y):
        count_not_empty = 0
        count_empty = 0
        
        x1      = round((x*100-1)/100,2)
        x2      = round((x*100+1)/100,2)
        y1      = round((y*100-1)/100,2)
        y2      = round((y*100+1)/100,2)
        
        v = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
        
        n = np.size(v,0)
        if x > 0 and y > 0 and (x1,y1) in self.empty_nodes:
            if (x,y) == (0.01,0.01):
                count_not_empty += 1
                del v[n-1]
            else:
                count_empty += 1
                v[count_empty-1] = (x1,y1)
        else:
            count_not_empty += 1
            del v[n-1]        
    
        n = np.size(v,0)
        if y > 0 and (x,y1) in self.empty_nodes:
            if (x,y) == (0,0.01) or (x,y) == (1,0.01):
                count_not_empty += 1
                del v[n-1]  
            else:
                count_empty += 1
                v[count_empty-1] = (x,y1)
        else:
            count_not_empty += 1
            del v[n-1]  
            
        n = np.size(v,0)
        if x < 1 and y > 0 and (x2,y1) in self.empty_nodes:
            if (x,y) == (0.99,0.01):
                count_not_empty += 1
                del v[n-1]  
            else:
                count_empty += 1
                v[count_empty-1] = (x2,y1)
        else:
            count_not_empty += 1
            del v[n-1]  
            
        n = np.size(v,0)
        if x > 0 and (x1,y) in self.empty_nodes:
            if (x,y) == (0.01,0) or (x,y) == (0.01,1):
                count_not_empty += 1
                del v[n-1]  
            else:
                count_empty += 1
                v[count_empty-1] = (x1,y)
        else:
            count_not_empty += 1
            del v[n-1]  
            
        n = np.size(v,0)
        if x < 1 and (x2,y) in self.empty_nodes:
            if (x,y) == (0.99,0) or (x,y) == (0.99,1):
                count_not_empty += 1
                del v[n-1]
            else:
                count_empty += 1
                v[count_empty-1] = (x2,y)
        else:
            count_not_empty += 1
            del v[n-1]  
            
        n = np.size(v,0)
        if x > 0 and y < 1 and (x1,y2) in self.empty_nodes:
            if (x,y) == (0.01,0.99):
                count_not_empty += 1
                del v[n-1]  
            else:
                count_empty += 1
                v[count_empty-1] = (x1,y2)
        else:
            count_not_empty += 1
            del v[n-1]  
            
        n = np.size(v,0)
        if y < 1 and (x,y2) in self.empty_nodes:
            if(x,y) == (0,0.99) or (x,y) == (1,0.99):
                count_not_empty += 1
                del v[n-1]
            else:
                count_empty += 1
                v[count_empty-1] = (x,y2)
        else:
            count_not_empty += 1
            del v[n-1]  
            
        n = np.size(v,0)
        if x < 1 and y < 1 and (x2,y2) in self.empty_nodes:
            if (x,y) == (0.99,0.99):
                count_not_empty += 1
                del v[n-1]
            else:
                count_empty += 1
                v[count_empty-1] = (x2,y2)
        else:
            count_not_empty += 1
            del v[n-1]  
                
        n = np.size(v,0)
        if 0< count_empty < n:
            v = v[0:count_empty]
            
        if count_empty == 0:
            return False
        else:
            return v    
    
    """dipol directions"""
    def get_dipole_direction_point_source(self):
        n1list = []
        n2list = []
        clist = []
        n1xlist = []
        n1ylist = []
        n2xlist = []
        n2ylist = []
        ndipole = []
        traction = self.traction
        for agent in self.agents:
            x = agent[0]
            y = agent[1]
            dx = 1/100
            x1=round(x-dx,2)
            x2=round(x+dx,2)
            y1=round(y-dx,2)
            y2=round(y+dx,2)
            h = self.agents[agent]

            if h == 1: #"""vertical"""
                node1_x = (Point(x,y1),0)
                node1_y = (Point(x,y1),traction)
                node2_x = (Point(x,y2),0)
                node2_y = (Point(x,y2),-traction)
                node1   = (x,y1)
                node2   = (x,y2)      
            elif h == 2: #"""horizontal"""
                node1_x = (Point(x1,y),traction)
                node1_y = (Point(x1,y),0)
                node2_x = (Point(x2,y),-traction)
                node2_y = (Point(x2,y),0)
                node1   = (x1,y)
                node2   = (x2,y)             
            elif h == 3: #"""diagonal from down left to up right"""
                node1_x = (Point(x1,y1),traction/sqrt(2))
                node1_y = (Point(x1,y1),traction/sqrt(2))
                node2_x = (Point(x2,y2),-traction/sqrt(2))
                node2_y = (Point(x2,y2),-traction/sqrt(2))
                node1   = (x1,y1)
                node2   = (x2,y2)     
            elif h == 4: #"""diagonal from up left to down right"""
                node1_x = (Point(x2,y1),-traction/sqrt(2))
                node1_y = (Point(x2,y1),traction/sqrt(2))
                node2_x = (Point(x1,y2),traction/sqrt(2))
                node2_y = (Point(x1,y2),-traction/sqrt(2))
                node1   = (x2,y1)
                node2   = (x1,y2) 

            n1list.append(node1)
            n2list.append(node2)
            clist.append((x,y))
            n1xlist.append(node1_x)
            n1ylist.append(node1_y)
            n2xlist.append(node2_x)
            n2ylist.append(node2_y)
            ndipole.append(h)

        nlist = (n1list,n2list,clist,n1xlist,n1ylist,n2xlist,n2ylist,ndipole)
        
        return nlist
        """dipol directions"""
    def set_dipole_direction_point_source(self,n_agent,xc,yc,hc):
        n1list = []
        n2list = []
        clist = []
        n1xlist = []
        n1ylist = []
        n2xlist = []
        n2ylist = []
        ndipole = []
        traction = self.traction
        for k in range(n_agent):
            x = xc[k]
            y = yc[k]
            dx = 1/100
            x1=round(x-dx,2)
            x2=round(x+dx,2)
            y1=round(y-dx,2)
            y2=round(y+dx,2)
            h = hc[k]

            if h == 1: #"""vertical"""
                node1_x = (Point(x,y1),0)
                node1_y = (Point(x,y1),traction)
                node2_x = (Point(x,y2),0)
                node2_y = (Point(x,y2),-traction)
                node1   = (x,y1)
                node2   = (x,y2)      
            elif h == 2: #"""horizontal"""
                node1_x = (Point(x1,y),traction)
                node1_y = (Point(x1,y),0)
                node2_x = (Point(x2,y),-traction)
                node2_y = (Point(x2,y),0)
                node1   = (x1,y)
                node2   = (x2,y)             
            elif h == 3: #"""diagonal from down left to up right"""
                node1_x = (Point(x1,y1),traction/sqrt(2))
                node1_y = (Point(x1,y1),traction/sqrt(2))
                node2_x = (Point(x2,y2),-traction/sqrt(2))
                node2_y = (Point(x2,y2),-traction/sqrt(2))
                node1   = (x1,y1)
                node2   = (x2,y2)     
            elif h == 4: #"""diagonal from up left to down right"""
                node1_x = (Point(x2,y1),-traction/sqrt(2))
                node1_y = (Point(x2,y1),traction/sqrt(2))
                node2_x = (Point(x1,y2),traction/sqrt(2))
                node2_y = (Point(x1,y2),-traction/sqrt(2))
                node1   = (x2,y1)
                node2   = (x1,y2) 

            n1list.append(node1)
            n2list.append(node2)
            clist.append((x,y))
            n1xlist.append(node1_x)
            n1ylist.append(node1_y)
            n2xlist.append(node2_x)
            n2ylist.append(node2_y)
            ndipole.append(h)
            
            # tol = 1e-4
            # if node1[0] < -tol or node1[1] < -tol or node2[0] < -tol or node2[1] < -tol:
            #     print(a,h)

        nlist = (n1list,n2list,clist,n1xlist,n1ylist,n2xlist,n2ylist,ndipole)
        
        return nlist
    """move cells"""
    def move(self):
        move_list = []
        self.old_agents = copy.deepcopy(self.agents)
        for agent in self.old_agents: 
            nn = randint(1,5)
            if nn == 1 or nn == 2 or nn == 3 or nn == 4:    
                if self.neighbour_empty(agent[0], agent[1]):
                    q = self.neighbour_empty(agent[0], agent[1])
                    empty_node = random.choice(q)
                    h = randint(1, 4) #4
                    x = empty_node[0]
                    y = empty_node[1]
                    tol = 1e-4
                    if abs(x) < tol or abs(x-1) < tol: #"""vertical""" 
                        h = 1 
                    elif abs(y) < tol or abs(y-1) < tol: #"""horizontal""" 
                        h = 2
                    old_h = self.agents[agent]
                    move_list.append((agent,old_h,empty_node,h))
                    self.agents[empty_node] = h
                    del self.agents[agent]
                    self.empty_nodes.remove(empty_node)
                    self.empty_nodes.append(agent)
        return move_list
            
    
    """determine deformation at cell position"""    
    def deformation(self,dis):
        deform = {}
        for agent in self.agents:
            x = agent[0]
            y = agent[1]
            dipole = self.agents[agent]            
            if dipole == 1: #"""vertical"""
                deform1 = abs(dis(x,round((y*100-1)/100,2))[1]-dis(x,round((y*100+1)/100,2))[1])
            elif dipole == 2: #"""horizontal"""
                deform1 = abs(dis(round((x*100-1)/100,2),y)[0]-dis(round((x*100+1)/100,2),y)[0])
            elif dipole == 3: #"""diagonal from down left to up right"""
                deform1 = sqrt((dis(round((x*100-1)/100,2),round((y*100-1)/100,2))[0]-dis(round((x*100+1)/100,2),round((y*100+1)/100,2))[0])**2+(dis(round((x*100-1)/100,2),round((y*100-1)/100,2))[1]-dis(round((x*100+1)/100,2),round((y*100+1)/100,2))[1])**2)
                # deform1 = pow(2,-1/6)*sqrt((dis(round((x*100-1)/100,2),round((y*100-1)/100,2))[0]-dis(round((x*100+1)/100,2),round((y*100+1)/100,2))[0])**2+(dis(round((x*100-1)/100,2),round((y*100-1)/100,2))[1]-dis(round((x*100+1)/100,2),round((y*100+1)/100,2))[1])**2)
            elif dipole == 4: #"""diagonal from up left to down right"""
                deform1 = sqrt((dis(round((x*100-1)/100,2),round((y*100+1)/100,2))[0]-dis(round((x*100+1)/100,2),round((y*100-1)/100,2))[0])**2+(dis(round((x*100-1)/100,2),round((y*100+1)/100,2))[1]-dis(round((x*100+1)/100,2),round((y*100-1)/100,2))[1])**2)
            deform[agent] = deform1
        return deform
    
    """compare the deformation at previous and new cell position and move back if needed"""
    def update_agents(self,deform,deform_new,move_list):
        for i in move_list:
            (old_agent,old_h,agent,h) = i
            if deform_new[agent] > deform[old_agent] and old_agent in self.empty_nodes:
                self.agents[old_agent] = old_h
                del self.agents[agent]
                self.empty_nodes.remove(old_agent)
                self.empty_nodes.append(agent)  
                
    """determine deformation at cell position"""    
    def manual_deformation(self,n_dipole,xc,yc,hc,dis):
        deform = []
        for k in range(n_dipole):
            x = xc[k]
            y = yc[k]
            dipole = hc[k]
            if dipole == 1: #"""vertical"""
                deform1 = abs(dis(x,round((y*100-1)/100,2))[1]-dis(x,round((y*100+1)/100,2))[1])
            elif dipole == 2: #"""horizontal"""
                deform1 = abs(dis(round((x*100-1)/100,2),y)[0]-dis(round((x*100+1)/100,2),y)[0])
            elif dipole == 3: #"""diagonal from down left to up right"""
                deform1 = sqrt((dis(round((x*100-1)/100,2),round((y*100-1)/100,2))[0]-dis(round((x*100+1)/100,2),round((y*100+1)/100,2))[0])**2+(dis(round((x*100-1)/100,2),round((y*100-1)/100,2))[1]-dis(round((x*100+1)/100,2),round((y*100+1)/100,2))[1])**2)
            elif dipole == 4: #"""diagonal from up left to down right"""
                deform1 = sqrt((dis(round((x*100-1)/100,2),round((y*100+1)/100,2))[0]-dis(round((x*100+1)/100,2),round((y*100-1)/100,2))[0])**2+(dis(round((x*100-1)/100,2),round((y*100+1)/100,2))[1]-dis(round((x*100+1)/100,2),round((y*100-1)/100,2))[1])**2)
            deform.append(deform1)
        return deform

def save_complete_state(fname,mesh,q,abm,t):
    save_state(mesh,q,fname,t)
    abm.save_state(fname)

def load_complete_state(fname,abm_pars):
    width       = abm_pars["width"]
    height      = abm_pars["height"]
    Ncell       = abm_pars["Ncell"]
    empty_ratio = abm_pars["empty_ratio"]
    traction    = abm_pars["traction"]
    case        = abm_pars["case"]
    planestrain = abm_pars["planestrain"]
    
    abm         = ABM(width, height,empty_ratio,traction)             
    mesh,q,W    = load_state(fname,case,planestrain)
    abm.load_state(fname)
    return abm,mesh,q,W                
           
# =========================================================================== #       
def outvtk_line(fname,p_list,u_displacement):
   f = open(fname + '.vtk', "w")
   f.write('# vtk DataFile Version 2.0\n')
   f.write('Agent list created from FEniCS\n')
   f.write('ASCII\n')
   f.write('DATASET UNSTRUCTURED_GRID\n')
   f.write('POINTS '+str(len(p_list))+' float\n')
   for p in p_list:
       p_set = Point(p) + Point(u_displacement(Point(p)))
       f.write(str(p_set[0])+' '+str(p_set[1])+' 0\n')
   f.close()
# =========================================================================== #     
def outvtk_agents(fname,p1l,p2l,u_displacement):
    ff = open(fname + '.vtk', 'w')
    ff.write('# vtk DataFile Version 2.0\n')
    ff.write('Agent list created from FEniCS\n')
    ff.write('ASCII\n')
    ff.write('DATASET POLYDATA\n')
    npoint = np.size(p1l,0)
    ff.write('POINTS '+str(2*npoint)+' float\n')
    for i in range(0,npoint):
        p1 = Point(p1l[i][0],p1l[i][1])
        p2 = Point(p2l[i][0],p2l[i][1])
        p_set1 = p1 + Point(u_displacement(p1))
        p_set2 = p2 + Point(u_displacement(p2))
        ff.write(str(p_set1[0])+' '+str(p_set1[1])+' 0\n')
        ff.write(str(p_set2[0])+' '+str(p_set2[1])+' 0\n')
    ff.write('POLYGONS '+str(npoint)+' '+str(3*npoint)+'\n') 
    k=0
    for i in range(0,npoint):
        ff.write('2 '+str(k)+' '+str(k+1)+'\n')
        k+=2
    ff.close()
