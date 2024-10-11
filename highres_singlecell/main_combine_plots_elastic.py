import numpy as np
import matplotlib.pyplot as plt
from fenics import *

from abm_module import *
#from os import mkdir

plt.rcParams['font.size'] = '7'
plt.rcParams["font.family"] = "Times New Roman"

folderselastic =   ['output_elastic_gent/','output_elastic_gent_plane_strain/','output_elastic_NH/','output_elastic_NH_plane_strain/']

for folder in folderselastic:
    
    abm_pars = read_dictionary(folder + 'run_pars.json')
    planestrain   = abm_pars["planestrain"]
    
    if folder == 'output_elastic_gent/' or folder == 'output_elastic_gent_plane_strain/':
        if planestrain == True:
            Label = 'Gent plane: strain case'     
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='k',linestyle='dashed')
        else:
            Label = 'Gent plane: stress case'
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='k',linestyle='-')
    else:
        if planestrain == True:
            Label = 'Neo-Hooke: plane strain case'     
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='gray',linestyle='dashed')
        else:
            Label = 'Neo-Hooke: plane stress case'
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='gray',linestyle='-')

    
plt.title('stress-strain relation pure elastic cases')
plt.legend()
plt.xlabel('strain')
plt.ylabel('force')
plt.ylim([0,7.5])
plt.xlim([0,0.89])

plt.savefig('forces_elastic.pdf',dpi=1200)