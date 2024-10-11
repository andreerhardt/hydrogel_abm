from   fenics import *
import pickle
import itertools
import random 
from random import randint
import numpy as np
import copy
import json

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
def output_old(file1,q, t=0):
    # The idea here is to create new mesh and deform this one for the purpose
    # of saving. So u and its mesh stay the same
    V = q.function_space()
    mesh = V.mesh()

    mesh_out = Mesh(mesh)  # Now we have a copy
    # Want to setup function space on this mesh ...
    Vout = FunctionSpace(mesh_out, V.ufl_element())
    q_out = Function(Vout)
    # ... and represent the displacement in it
    q_out.vector().axpy(1, q.vector())
    
    u_out,*_= q_out.split()
    # Deform
    ALE.move(mesh_out,q_out.sub(0))
    u_out.rename('u','Displacement')
   
    #file1 << (u_out, t)
    return u_out, mesh_out

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
