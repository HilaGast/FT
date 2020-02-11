
from dipy.tracking.utils import length
from dipy.tracking.streamline import Streamlines, cluster_confidence
from FT.single_fascicle_vizualization import *

from dipy.viz import window, actor
from FT.all_subj import all_subj_folders, all_subj_names
import numpy as np
from FT.weighted_tracts import *

main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'
folder_name = main_folder + all_subj_folders[0]
n = all_subj_names[0]
nii_file = load_dwi_files(folder_name)[5]

tract_path = folder_name + r'\streamlines' + n + '_wholebrain_3d_plus_new.trk'

streamlines = load_ft(tract_path, nii_file)

lab_labels_index, affine = nodes_by_index_mega(folder_name)
masked_streamlines = choose_specific_bundle(streamlines, affine, folder_name, mask_type='cc')
streamline_dict = create_streamline_dict(masked_streamlines, lab_labels_index, affine)

# streamline_dict = clean_non_cc(streamline_dict) ##

mat_medians = load_mat_of_median_vals(mat_type='w_plus')
index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlas2nii.txt'
idx = nodes_labels_mega(index_to_text_file)[1]
id = np.argsort(idx)
mat_medians = mat_medians[id]
mat_medians = mat_medians[:, id]
vec_vols = []
s_list = []
'''new func:'''
for i in range(id.__len__()):  #
    for j in range(i + 1):  #
        edge_s_list = []
        # print(i,j)
        if (i + 1, j + 1) in streamline_dict and mat_medians[i, j] > 0:
            edge_s_list += streamline_dict[(i + 1, j + 1)]
        if (j + 1, i + 1) in streamline_dict and mat_medians[i, j] > 0:
            edge_s_list += streamline_dict[(j + 1, i + 1)]
        edge_vec_vols = [mat_medians[i, j]] * edge_s_list.__len__()

        s_list = s_list + edge_s_list
        vec_vols = vec_vols + edge_vec_vols

s = Streamlines(s_list)

cci = cluster_confidence(s)


keep_streamlines = Streamlines()
for i, sl in enumerate(s):
    if cci[i] >= 1:
        keep_streamlines.append(sl)

# Visualize the streamlines we kept
ren = window.Renderer()

keep_streamlines_actor = actor.line(keep_streamlines, linewidth=0.1)

ren.add(keep_streamlines_actor)


interactive = True
if interactive:
    window.show(ren)

