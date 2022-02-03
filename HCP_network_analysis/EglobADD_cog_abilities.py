import os, glob
import numpy as np
from network_analysis.global_network_properties import get_efficiency
from draw_scatter_fit import draw_scatter_fit
import pandas as pd

shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')
add_eff=[]
num_eff=[]
total=[]
fluid=[]
crystal=[]
table1 = pd.read_csv('F:\data\V7\HCP\HCP_behavioural_data.csv')
cerbellum_i = [i for i in range(123,151)]

for sl in shortlist:
    #subjnum = str.split(sl, os.sep)[2]
    #dir_name = f'F:\data\V7\HCP{os.sep}{subjnum}'
    dir_name = sl[:-1]
    subj_number = sl.split(os.sep)[-2]

    add_cm = np.load(f'{sl}cm{os.sep}add_bna_cm_ord.npy')
    add_cm = np.delete(add_cm, cerbellum_i, axis=0)
    add_cm = np.delete(add_cm, cerbellum_i, axis=1)

    num_cm = np.load(f'{sl}cm{os.sep}num_bna_cm_ord.npy')
    num_cm = np.delete(num_cm, cerbellum_i, axis=0)
    num_cm = np.delete(num_cm, cerbellum_i, axis=1)

    #add_cm = np.load(f'{sl}cm_add.npy')
    #num_cm = np.load(f'{sl}cm_num.npy')


    add_eff.append(get_efficiency(add_cm))

    num_eff.append(get_efficiency(num_cm))

    total.append(float(table1['CogTotalComp_AgeAdj'][table1['Subject']==int(subj_number)].values))
    fluid.append(float(table1['CogFluidComp_AgeAdj'][table1['Subject']==int(subj_number)].values))
    crystal.append(float(table1['CogCrystalComp_AgeAdj'][table1['Subject']==int(subj_number)].values))

#subj2_remove = [21,40,60,62,96,107]
#subj2_remove = [39,94]
#subj2_remove = [2,14,20,39,42,59,61,70,94,105,106,107,113]
# num_eff = np.asarray(num_eff)
# num_eff[subj2_remove] = np.nan
#
# add_eff = np.asarray(add_eff)
# add_eff[subj2_remove] = np.nan

draw_scatter_fit(num_eff,add_eff, comp_reg=True, ttl='Eglob ADD vs Eglob Num',norm_x=False)
draw_scatter_fit(total, add_eff, ttl='Eadd total', comp_reg=True,norm_x=False)
draw_scatter_fit(fluid, add_eff, ttl='Eadd fluid', comp_reg=True,norm_x=False)
draw_scatter_fit(crystal, add_eff, ttl='Eadd crystal', comp_reg=True,norm_x=False)

draw_scatter_fit(total, num_eff, ttl='Enum total', comp_reg=True,norm_x=False)
draw_scatter_fit(fluid, num_eff, ttl='Enum fluid', comp_reg=True,norm_x=False)
draw_scatter_fit(crystal, num_eff, ttl='Enum crystal', comp_reg=True,norm_x=False)
