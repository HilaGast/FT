from time_delay_index.boxplot_hcp_tdi_subnetworks import compare_networks, boxplot_8networks, keep_one_hemi
from Tractography.group_analysis import create_all_subject_connectivity_matrices, norm_matrices
import glob, os
from yeo_sub_networks.from_wb_mat_to_subs import from_whole_brain_to_networks

mat_type = 'time_th3'
atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
atlas = 'yeo7_200'
subjects = glob.glob(f'F:\Hila\TDI\siemens\C*{os.sep}D60d11{os.sep}cm{os.sep}{mat_type}_{atlas}_cm_ord.npy')
connectivity_matrices = create_all_subject_connectivity_matrices(subjects)
norm_cm = norm_matrices(connectivity_matrices, norm_type='scaling')
networks_matrices, network_mask_vecs = from_whole_brain_to_networks(norm_cm, atlas_index_labels,
                                                                    hemi_flag=True)
networks_matrices = keep_one_hemi(networks_matrices, 'LH')
network_names = list(networks_matrices.keys())
val_per_net = compare_networks(networks_matrices, network_names)
boxplot_8networks(val_per_net, hemi='LH')
