import numpy as np
import matplotlib.pyplot as plt
from fenics import *

from abm_module import *

plt.rcParams['font.size'] = '7'
plt.rcParams["font.family"] = "Times New Roman"

foldersg =   ['output_elastic_gent/','output_elastic_gent_plane_strain/','output_hydro_gent_CH_0005/','output_hydro_gent_CH_05/','output_hydro_gent_CH_50/','output_hydro_gent_CH_0005_plane_strain/','output_hydro_gent_CH_05_plane_strain/','output_hydro_gent_CH_50_plane_strain/']

colormap = plt.cm.nipy_spectral
colors = [colormap(i) for i in np.linspace(0, 1, len(foldersg))]
COLOR =["tab:blue","tab:orange","tab:green","tab:red"]
i = 0
k = 0

fig, ax = plt.subplots(figsize=(8/3,2))

for folder in foldersg:
    
    abm_pars = read_dictionary(folder + 'run_pars.json')
    planestrain   = abm_pars["planestrain"]
    AllenCahn     = abm_pars["AllenCahn"]
    
    if folder == 'output_elastic_gent/' or folder == 'output_elastic_gent_plane_strain/':
        if planestrain == True:
            Label = 'pure elastic plane strain'     
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='k',linestyle='dashed')
        else:
            Label = 'pure elastic plane stress'
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='k',linestyle='-')
    else:
        eps           = abm_pars["eps"]
        C0            = abm_pars["C0"]
        
        if AllenCahn == True:
            if planestrain == True:
                Label = 'plane strain, $k=$'+str(C0/2)
            else:
                Label = 'plane stress, $k=$'+str(C0/2)
        else:
            if planestrain == True:
                Label = 'plane strain, $k=$'+str(C0/2)
            else:
                Label = 'plane stress, $k=$'+str(C0/2)
        if planestrain == True:
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=0.5,linestyle='dashed',color = COLOR[i])
            i +=1
        else:
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=0.5,linestyle='-',color = COLOR[k])
            k += 1
    
plt.legend(fontsize=5)
plt.xlabel('strain $\epsilon_\mathrm{eng}$')
plt.ylabel('stress $\sigma_\mathrm{eng}$')
plt.ylim([0,7.5])
plt.xlim([0,1])

plt.savefig('forces_CH.pdf',dpi=1200, bbox_inches='tight',pad_inches = 0)
# plt.show()
