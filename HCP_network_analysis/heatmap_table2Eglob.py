import os, glob
import pandas as pd
from network_analysis.global_network_properties import get_efficiency
from calc_corr_statistics.pearson_r_calc import *

shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')

subj = []
sn = []
add_eff=[]
num_eff=[]
for sl in shortlist:
    if os.path.exists(sl+'rBN_Atlas_274_combined_1mm.nii'):
        subj.append(sl)
        sn.append(int(sl.split(os.sep)[-2]))
        add_cm = np.load(f'{sl}cm_add.npy')
        add_eff.append(get_efficiency(add_cm))

        num_cm = np.load(f'{sl}cm_num.npy')
        num_eff.append(get_efficiency(num_cm))

table1 = pd.read_csv('F:\data\V7\HCP\HCP_behavioural_data.csv')
table2 = table1.loc[table1['Subject'].isin(sn)]

num_table2 = table2.select_dtypes(include=['int16','int32','int64','float16','float32','float64'])

r = np.zeros((2,len(num_table2.columns)))
p = np.zeros((2,len(num_table2.columns)))
for i,c in enumerate(num_table2.columns):

    y = list(num_table2[c])
    r_add,p_add = calc_corr(add_eff,y)
    r_num,p_num = calc_corr(num_eff,y)

    r[0,i] = r_num
    p[0,i] = p_num
    r[1,i] = r_add
    p[1,i] = p_add

tabler = pd.DataFrame(data=r, index=['num','add'], columns=num_table2.columns)