import numpy as np
import matplotlib.pyplot as plt
from fenics import *

from abm_module import *

plt.rcParams['font.size'] = '7'
plt.rcParams["font.family"] = "Times New Roman"

folders =   ['output_elastic_gent/','output_hydro_gent_AC_0005/','output_hydro_gent_AC_05/','output_hydro_gent_AC_50/','output_elastic_NH/','output_hydro_NH_AC_0005/','output_hydro_NH_AC_05/','output_hydro_NH_AC_50/']


COLOR =["tab:blue","tab:orange","tab:green","tab:red"]
i = 0
k = 0

fig, ax = plt.subplots(figsize=(8/3,2))

for folder in folders:
    
    abm_pars = read_dictionary(folder + 'run_pars.json')
    case          = abm_pars["case"]
    AllenCahn     = abm_pars["AllenCahn"]
    
    if folder == 'output_elastic_gent/' or folder == 'output_elastic_NH/':
        if case == 1:
            Label = 'pure elastic neo-Hooke'     
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='k',linestyle='dashed')
        else:
            Label = 'pure elastic Gent'
            E = np.load(folder+'energies.npy')
            plt.plot(E[1],E[3],label=Label,linewidth=1.0,color='k',linestyle='-')
    else:
        eps           = abm_pars["eps"]
        C0            = abm_pars["C0"]
        
        if case == 3:
            Label = 'Neo-Hooke hydrogel, $k=$'+str(C0/2)
        else:
            Label = 'Gent hydrogel, $k=$'+str(C0/2)

        if case == 3:
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
plt.ylim([0,6])
plt.xlim([0,1])

plt.savefig('forces_AC.pdf',dpi=1200, bbox_inches='tight',pad_inches = 0)
