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

for sl in shortlist:
    #subjnum = str.split(sl, os.sep)[2]
    #dir_name = f'F:\data\V7\HCP{os.sep}{subjnum}'
    dir_name = sl[:-1]
    subj_number = sl.split(os.sep)[-2]



    add_cm = np.load(f'{dir_name}{os.sep}cm_add.npy')
    add_eff.append(get_efficiency(add_cm))

    num_cm = np.load(f'{dir_name}{os.sep}cm_num.npy')
    num_eff.append(get_efficiency(num_cm))

    total.append(float(table1['CogTotalComp_AgeAdj'][table1['Subject']==int(subj_number)].values))
    fluid.append(float(table1['CogFluidComp_AgeAdj'][table1['Subject']==int(subj_number)].values))
    crystal.append(float(table1['CogCrystalComp_AgeAdj'][table1['Subject']==int(subj_number)].values))

subj2_remove = [21,40,60,62,96,107]
num_eff = np.asarray(num_eff)
num_eff[subj2_remove] = np.nan

add_eff = np.asarray(add_eff)
add_eff[subj2_remove] = np.nan

draw_scatter_fit(num_eff,add_eff, comp_reg=True, ttl='Eglob ADD vs Eglob Num',norm_x=False)
draw_scatter_fit(total, add_eff, ttl='Eadd total', comp_reg=True,norm_x=False)
draw_scatter_fit(fluid, add_eff, ttl='Eadd fluid', comp_reg=True,norm_x=False)
draw_scatter_fit(crystal, add_eff, ttl='Eadd crystal', comp_reg=True,norm_x=False)

draw_scatter_fit(total, num_eff, ttl='Enum total', comp_reg=True,norm_x=False)
draw_scatter_fit(fluid, num_eff, ttl='Enum fluid', comp_reg=True,norm_x=False)
draw_scatter_fit(crystal, num_eff, ttl='Enum crystal', comp_reg=True,norm_x=False)
