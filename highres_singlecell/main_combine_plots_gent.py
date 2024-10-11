import numpy as np
import matplotlib.pyplot as plt
from fenics import *

from abm_module import *
#from os import mkdir

plt.rcParams['font.size'] = '6'
plt.rcParams["font.family"] = "Times New Roman"

foldersg =   ['output_elastic_gent/','output_elastic_gent_plane_strain/','output_hydro_gent_AC_0005/','output_hydro_gent_AC_05/','output_hydro_gent_AC_50/','output_hydro_gent_CH_0005/','output_hydro_gent_CH_05/','output_hydro_gent_CH_50/','output_hydro_gent_AC_0005_plane_strain/','output_hydro_gent_AC_05_plane_strain/','output_hydro_gent_AC_50_plane_strain/','output_hydro_gent_CH_0005_plane_strain/','output_hydro_gent_CH_05_plane_strain/','output_hydro_gent_CH_50_plane_strain/']
# foldersg =   ['output_elastic_gent/','output_hydro_gent_AC_05/','output_hydro_gent_AC_50/','output_hydro_gent_CH_05/','output_hydro_gent_CH_50/']

colormap = plt.cm.nipy_spectral#gist_ncar #nipy_spectral, Set1,Paired   
colors = [colormap(i) for i in np.linspace(0, 1, len(foldersg))]

for folder in foldersg:
    
    abm_pars = read_dictionary(folder + 'run_pars.json')
    planestrain   = abm_pars["planestrain"]
    AllenCahn     = abm_pars["AllenCahn"]
    
    if folder == 'output_elastic_gent/' or folder == 'output_elastic_gent_plane_strain/':
        if planestrain == True:
            Label = 'pure elastic plane strain case'     
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='k',linestyle='dashed')
        else:
            Label = 'pure elastic plane stress case'
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='k',linestyle='-')
    else:
        eps           = abm_pars["eps"]
        C0            = abm_pars["C0"]
        
        if AllenCahn == True:
            if planestrain == True:
                Label = 'AC plane strain hydrogel with $k=$'+str(C0/2)+', $\epsilon=$'+str(eps)
            else:
                Label = 'AC plane stress hydrogel with $k=$'+str(C0/2)+', $\epsilon=$'+str(eps)
        else:
            if planestrain == True:
                Label = 'CH plane strain hydrogel with $k=$'+str(C0/2)+', $\epsilon=$'+str(eps)
            else:
                Label = 'CH plane stress hydrogel with $k=$'+str(C0/2)+', $\epsilon=$'+str(eps)
        if planestrain == True:
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=0.5,linestyle='dashed')
        else:
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=0.5,linestyle='-')
    
plt.title('stress-strain relation Gent models')
plt.legend()
plt.xlabel('strain')
plt.ylabel('force')
plt.ylim([0,7.5])
plt.xlim([0,0.89])

plt.savefig('forces.pdf',dpi=1200)
# plt.show()