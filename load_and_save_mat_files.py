from all_subj import all_subj_names,all_subj_folders,index_to_text_file,subj_folder
import numpy as np
from weighted_tracts import nodes_labels_yeo7
import scipy.io as sio

subj = all_subj_folders
names = all_subj_names
i=0
for s, n in zip(subj, names):
    i+=1
    idx = nodes_labels_yeo7(index_to_text_file)[1]
    id = np.argsort(idx)

    folder_name = subj_folder + s
    axcaliber_file = rf'{folder_name}\weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
    axcaliber_mat = np.load(axcaliber_file)
    axcaliber_mat = axcaliber_mat[id]
    axcaliber_mat = axcaliber_mat[:, id]

    num_of_tracts_file = rf'{folder_name}\non-weighted_wholebrain_5d_labmask_yeo7_200_nonnorm.npy'
    num_of_tracts_mat = np.load(num_of_tracts_file)
    num_of_tracts_mat = np.asarray(num_of_tracts_mat,dtype='float64')
    num_of_tracts_mat = num_of_tracts_mat[id]
    num_of_tracts_mat = num_of_tracts_mat[:, id]


    fa_file = rf'{folder_name}\weighted_wholebrain_5d_labmask_yeo7_200_FA_nonnorm.npy'
    fa_mat = np.load(fa_file)
    fa_mat = fa_mat / 100
    fa_mat = fa_mat[id]
    fa_mat = fa_mat[:, id]

    mat_file_name = rf'{folder_name}\subj{i}.mat'
    sio.savemat(mat_file_name, {'axcaliber': axcaliber_mat,'number_of_tracts':num_of_tracts_mat,'fa':fa_mat})
