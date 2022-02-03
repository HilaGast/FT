import os, glob
import pandas as pd
from draw_scatter_fit import *
from calc_corr_statistics.pearson_r_calc import *
from parcellation.group_weight import weight_atlas_by_add, save_as_nii_aal

shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')

total=[]
fluid=[]
crystal=[]
add=[]

table1 = pd.read_csv('F:\data\V7\HCP\HCP_behavioural_data.csv')
cerbellum_i = [i for i in range(123,151)]


for sl in shortlist:
    dir_name = sl[:-1]
    subj_number = sl.split(os.sep)[-2]

    add_cm = np.load(f'{sl}cm{os.sep}add_bna_cm_ord.npy')
    add_cm = np.delete(add_cm, cerbellum_i, axis=0)
    add_cm = np.delete(add_cm, cerbellum_i, axis=1)

    num_cm = np.load(f'{sl}cm{os.sep}num_bna_cm_ord_corrected.npy')
    num_cm = np.delete(num_cm, cerbellum_i, axis=0)
    num_cm = np.delete(num_cm, cerbellum_i, axis=1)
    mutual = np.nansum(add_cm*num_cm)

    add.append(mutual/np.nansum(num_cm))

    total.append(float(table1['CogTotalComp_AgeAdj'][table1['Subject']==int(subj_number)].values))
    fluid.append(float(table1['CogFluidComp_AgeAdj'][table1['Subject']==int(subj_number)].values))
    crystal.append(float(table1['CogCrystalComp_AgeAdj'][table1['Subject']==int(subj_number)].values))

draw_scatter_fit(total, add, ttl='total', comp_reg=True,norm_x=False)
draw_scatter_fit(fluid, add, ttl='fluid', comp_reg=True,norm_x=False)
draw_scatter_fit(crystal, add, ttl='crystal', comp_reg=True,norm_x=False)