from balance.ADD_before_vs_after import *
from calc_corr_statistics.pearson_r_calc import *
from balance.delta_ADD_vs_delta_RMS import load_rms_subj
from HCP_network_analysis.prediction_model.predict_traits_by_networks import from_whole_brain_to_networks
def create_all_subject_connectivity_matrices(subjects, atlas, weight):
    connectivity_matrices = []
    for s in subjects:
        mat_name = f'{s}{os.sep}cm{os.sep}{weight}_{atlas}_cm_ord.npy'
        connectivity_matrices.append(np.load(mat_name))
    connectivity_matrices = np.array(connectivity_matrices)
    connectivity_matrices = np.swapaxes(connectivity_matrices, 0, -1)

    return connectivity_matrices


if __name__ == '__main__':
    weight ='ADD'
    main_folder = r'F:\Hila\balance'
    atlas_name = 'yeo7_200'
    atlas_main_folder = r'G:\data\atlases\yeo'
    atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'

    before_subj = glob.glob(main_folder + f'{os.sep}e*{os.sep}before{os.sep}*')
    after_subj = glob.glob(main_folder + f'{os.sep}e*{os.sep}after{os.sep}*')

    rms = load_rms_subj(before_subj)

    before_con_mats = create_all_subject_connectivity_matrices(before_subj, atlas_name, weight)
    after_con_mats = create_all_subject_connectivity_matrices(after_subj, atlas_name, weight)
    delta_con_mats = after_con_mats-before_con_mats
    networks_matrices, network_mask_vecs = from_whole_brain_to_networks(delta_con_mats, atlas_index_labels)
    sommot_idx = network_mask_vecs['SomMot']
    sommot_matrices = networks_matrices['SomMot'].reshape(len(sommot_idx),-1)
    sommot_vals = sommot_matrices[sommot_idx,:]
    #before_vol_mat, mni_atlas_file_name, idx, subj_idx =  volume_based_var(atlas_name, weight, atlas_main_folder, before_subj)
    #after_vol_mat, mni_atlas_file_name, idx, subj_idx =  volume_based_var(atlas_name, weight, atlas_main_folder, after_subj)
    # delta_vol_mat = np.absolute(after_vol_mat-before_vol_mat)

    r, p = calc_corr(rms, delta_vol_mat, fdr_correct=False, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'deltaRMS_vs_deltaADD_r_th_'+atlas_name, main_folder)

    r, p = calc_corr(rms, delta_vol_mat, fdr_correct=True, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'deltaRMS_vs_deltaADD_r_th_fdr_'+atlas_name, main_folder)




