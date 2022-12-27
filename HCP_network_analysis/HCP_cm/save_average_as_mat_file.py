import scipy.io as sio
import os
import numpy as np

main_fol = 'G:\data\V7\HCP'
fol = f'{main_fol}{os.sep}cm{os.sep}'
atlases = ['yeo7_200']
ncm_options = ['SC']#, 'SPE']
weights = ['Num_Org', 'Num_HistMatch', 'FA_Org', 'FA_HistMatch', 'ADD_Org', 'ADD_HistMatch', 'Dist_Org',
           'Dist_HistMatch', 'NumxDist_HistMatch', 'NumxADD_HistMatch', 'ADDxDist_HistMatch']
for atlas in atlases:
    for ncm in ncm_options:
        for w in weights:
            idx = np.load(f'{fol}{atlas}_cm_ord_lookup.npy')
            id = np.argsort(idx)
            file_name = rf'{fol}average_{atlas}_{w}_{ncm}.npy'
            file_mat = np.asarray(np.load(file_name), dtype='float64')
            file_mat = file_mat[id]
            file_mat = file_mat[:, id]

            mat_file_name = rf'{fol}groupaverage_{atlas}_{w}_{ncm}.mat'
            sio.savemat(mat_file_name, {'mat': file_mat})