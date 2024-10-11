import numpy as np
import matplotlib.pyplot as plt
from fenics import *

from abm_module import *

plt.rcParams['font.size'] = '7'
plt.rcParams["font.family"] = "Times New Roman"

folderselastic =   ['output_elastic_gent/','output_elastic_gent_plane_strain/','output_elastic_NH/','output_elastic_NH_plane_strain/']

fig, ax = plt.subplots(figsize=(8/3,2))

for folder in folderselastic:
    
    abm_pars = read_dictionary(folder + 'run_pars.json')
    planestrain   = abm_pars["planestrain"]
    
    if folder == 'output_elastic_gent/' or folder == 'output_elastic_gent_plane_strain/':
        if planestrain == True:
            Label = 'Gent: plane strain'     
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='k',linestyle='dashed')
        else:
            Label = 'Gent: plane stress'
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='k',linestyle='-')
    else:
        if planestrain == True:
            Label = 'Neo-Hooke: plane strain'     
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='gray',linestyle='dashed')
        else:
            Label = 'Neo-Hooke: plane stress'
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='gray',linestyle='-')

plt.legend(fontsize=5)
plt.xlabel('strain $\epsilon_\mathrm{eng}$')
plt.ylabel('stress $\sigma_\mathrm{eng}$')
plt.ylim([0,7.5])
plt.xlim([0,1])

plt.savefig('forces_elastic.pdf',dpi=1200, bbox_inches='tight',pad_inches = 0)
