import numpy as np
import pandas as pd
import os
num_subj = 999
atlas = 'yeo7_100'
loaded_conn_mat = np.loadtxt(r'G:\data\V7\HCP\fmri\Schaefer_100Parcels_Conn_Mats_HCP_999_subj.csv')
conn_mats = loaded_conn_mat.reshape(
    loaded_conn_mat.shape[0], loaded_conn_mat.shape[1] // num_subj, num_subj)

table = pd.read_csv(r'G:\data\V7\HCP\fmri\my_HCP_subj.csv')
subj_list = list(table['Subject'])
n=0
for i,s_name in enumerate(subj_list):
    if os.path.exists(rf'G:\data\V7\HCP\{s_name}'):
        n+=1
        fmri_cm = conn_mats[:,:,i]
        np.save(rf'G:\data\V7\HCP\{s_name}\cm\{atlas}_fmri_Org_SC_cm_ord.npy',fmri_cm)
print(n)