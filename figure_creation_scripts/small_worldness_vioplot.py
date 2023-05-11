import os, glob
import numpy as np

from cc_analysis.cc_boxplot import create_cc_vioplot
from network_analysis.global_network_properties import get_swi
import pandas as pd

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
add_swi=[]
num_swi=[]
fa_swi=[]

th = 'Org'
atlas = 'yeo7_200'
for sl in subj_list:

    add_cm = np.load(f'{sl}cm{os.sep}{atlas}_ADD_{th}_SC_cm_ord.npy')
    add_swi.append(get_swi(add_cm))


    num_cm = np.load(f'{sl}cm{os.sep}{atlas}_Num_{th}_SC_cm_ord.npy')
    num_swi.append(get_swi(num_cm))


    fa_cm = np.load(f'{sl}cm{os.sep}{atlas}_FA_{th}_SC_cm_ord.npy')
    fa_swi.append(get_swi(fa_cm))


table = pd.DataFrame({'Num':num_swi, 'FA':fa_swi, 'ADD':add_swi})
pd.DataFrame.to_excel(table, 'G:\data\V7\HCP\swp_table.xlsx')
create_cc_vioplot(table)