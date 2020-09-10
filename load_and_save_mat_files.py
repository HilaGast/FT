
import scipy.io as sio
# load:
mat_file_name = r''
var_name = ''
mat=sio.loadmat(mat_file_name)
var = mat[var_name]

#save:
''' saves "var" variable into a variable named "var_name" in "mat_file_name" .mat file'''
sio.savemat(mat_file_name,{'var_name':var})


n=1
for r in nwei:
    nwei_h=nwei_h+list(r[0:n])
    n+=1


from all_subj import all_subj_names,all_subj_folders,index_to_text_file,subj_folder
import numpy as np
from weighted_tracts import nodes_labels_aal3
import scipy.io as sio

subj = all_subj_folders
names = all_subj_names
i=0
for s, n in zip(subj, names):
    i+=1
    idx = nodes_labels_aal3(index_to_text_file)[1]
    id = np.argsort(idx)

    folder_name = subj_folder + s
    axcaliber_file = rf'{folder_name}\weighted_mega_wholebrain_4d_labmask_aal3_nonnorm.npy'
    axcaliber_mat = np.load(axcaliber_file)
    axcaliber_mat = axcaliber_mat[id]
    axcaliber_mat = axcaliber_mat[:, id]

    num_of_tracts_file = rf'{folder_name}\non-weighted_mega_wholebrain_4d_labmask_aal3_nonnorm.npy'
    num_of_tracts_mat = np.load(num_of_tracts_file)
    num_of_tracts_mat = num_of_tracts_mat[id]
    num_of_tracts_mat = num_of_tracts_mat[:, id]

    fa_file = rf'{folder_name}\weighted_mega_wholebrain_4d_labmask_aal3_FA_nonnorm.npy'
    fa_mat = np.load(fa_file)
    fa_mat = fa_mat[id]
    fa_mat = fa_mat[:, id]

    mat_file_name = rf'{folder_name}\subj{i}.mat'
    sio.savemat(mat_file_name, {'axcaliber': axcaliber_mat,'number_of_tracts':num_of_tracts_mat,'fa':fa_mat})
