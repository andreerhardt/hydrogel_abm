# This example provides some extra functionality on top of the existing 
# FEniCS functions refine and adapt and is suited, e.g., for heuristic 
# mesh refinement of phase field models with interfaces.
#
# Authors: Leonie Schmeller & Dirk Peschka

from fenics import *

# used for debugging indices and values
#def check_dofs(mesh,u):
#    print("size = ",u.vector().size())
#    dm = u.function_space().dofmap()
#    uv = u.vector()
#    for cell_no in range(mesh.num_cells()):
#        cd = dm.cell_dofs(cell_no)
#        print("cell=",cell_no," cell_dofs(cd)=",cd,'-uv[cd]',uv[cd],'-min uv[cd]=',uv[cd].min(),'-max uv[cd]=',uv[cd].max())

# function: mark_isolevel
#     For an existing "mesh" and an existing function "u" defined on that mesh,
#     this function returns a boolean MeshFunction (cell based) which is true for
#     cells that contain dof values both above and below an isolevel (with uncertainty
#     encoded in ceps), e.g., so that at least for P1 it is ensured that an isoline 
#     crosses this cell. Otherwise the MeshFunction is false for that cell. This 
#     MeshFunction can be used to trigger refinement. Other marker functions, e.g., 
#     based on measured for the error of solutions are possible / desired.
def mark_isolevel(mesh,u,iso_level=0.0,ceps=0.0):
    cell_markers = MeshFunction("bool",mesh,mesh.geometric_dimension())
    cell_markers.set_all(False)

    dm = u.function_space().dofmap()
    uv = u.vector()

    for cell in cells(mesh):
        cell_index = cell.index()
        cell_dofs = dm.cell_dofs(cell_index)
        umin = uv[cell_dofs].min()
        umax = uv[cell_dofs].max()
        if (umin<(iso_level+ceps)) and (umax>(iso_level-ceps)):
            # not sure if cell_markers[cell] or cell_markers[cell_index] makes any difference?!
            # cell_markers[cell] = True
            cell_markers[cell_index] = True
            
    return cell_markers

# function: mark_diff
#     For an existing "mesh" and an existing function "u" defined on that mesh,
#     this function returns a boolean MeshFunction (cell based) which is true for
#     cells, where the difference of max and min value on that cell exceed a certain
#     value delta. Otherwise the MeshFunction is false for that cell. This 
#     MeshFunction can be used to trigger refinement. 
def mark_diff(mesh,u,delta):
    cell_markers = MeshFunction("bool",mesh,mesh.geometric_dimension())
    cell_markers.set_all(False)
    dm = u.function_space().dofmap()   
    uv = u.vector() # .array()

    for cell in cells(mesh):
        cell_index = cell.index()
        cell_dofs = dm.cell_dofs(cell_index)
        umin = uv[cell_dofs].min()
        umax = uv[cell_dofs].max()
        if (umax - umin) > delta :
            cell_markers[cell] = True

    return cell_markers

# function: mark_collect
#     Logical or between two boolean mesh function used for refinement. This is useful
#     if multiple marker functions are used in a single refinement step.
def mark_collect(marker1,marker2):
  out_marker = MeshFunction("bool",mesh,mesh.geometric_dimension())
  for i in range(marker.size()):
      out_marker[i] = (marker1[i] or marker2[i])
  return out_marker

# function: refine_and_adapt
#        For a given cell marker, mesh and refinement level this function counts 
#        the refined cells by increments, creates the refined mesh. If the optional 
#        argument "level" is not provided, then it is created and initialized to {0,1}.
#        This has the advantage that, in conjunction with set_max_level some 
#        too-fine refinement can be prevented.
def refine_and_adapt(marker,mesh,level=None):
    level_coarse = MeshFunction("size_t", mesh, mesh.geometric_dimension())
    # level == None -> level_coarse = 1 on refined and 0 otherwise
    # level exists -> level_coarse = level + 1 on refined and level otherwise
    if (level==None):
      for i in range(marker.size()):
        if (marker[i] == True):
          level_coarse[i] = 1
        else:
          level_coarse[i] = 0
    else:
      for i in range(marker.size()):
        if (marker[i] == True):
            level_coarse[i] = level[i] + 1
        else:
            level_coarse[i] = level[i]
  
    mesh_fine  = refine(mesh, marker)
    level_fine = adapt(level_coarse,mesh_fine)
    return mesh_fine,level_fine

# function: set_max_level
#         Makes sure that the level cannot exceed a certain maximal level "max_level". I
#         marker[cell] = True but level[cell] = max_level, then marker[cell] != False 
def set_max_level(marker,level,max_level):
  for i in range(marker.size()):
    if (level[i]>=max_level):
      marker[i]=False
  return marker