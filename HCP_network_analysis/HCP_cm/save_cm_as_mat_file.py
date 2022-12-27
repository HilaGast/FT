import scipy.io as sio
import glob, os
import numpy as np

main_fol = 'G:\data\V7\HCP'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}cm{os.sep}')
th='_histmatch_th'
atlas='yeo7_200'
for fol in all_subj_fol:
    sn = fol.split(os.sep)[-3]

    idx = np.load(f'{fol}{atlas}_cm_ord_lookup.npy')
    id = np.argsort(idx)

    axsi_file = rf'{fol}\add_{atlas}_cm_ord{th}.npy'
    axsi_mat = np.asarray(np.load(axsi_file),dtype='float64')
    axsi_mat = axsi_mat[id]
    axsi_mat = axsi_mat[:, id]

    num_of_tracts_file = rf'{fol}\num_{atlas}_cm_ord{th}.npy'
    num_of_tracts_mat = np.asarray(np.load(num_of_tracts_file),dtype='float64')
    num_of_tracts_mat = num_of_tracts_mat[id]
    num_of_tracts_mat = num_of_tracts_mat[:, id]

    fa_file = rf'{fol}\fa_{atlas}_cm_ord{th}.npy'
    fa_mat = np.asarray(np.load(fa_file),dtype='float64')
    fa_mat = fa_mat[id]
    fa_mat = fa_mat[:, id]

    mat_file_name = rf'{fol}{sn}_{atlas}{th}.mat'
    sio.savemat(mat_file_name, {'axsi': axsi_mat,'number_of_tracts':num_of_tracts_mat,'fa':fa_mat})



