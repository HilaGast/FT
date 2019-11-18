from FT.weighted_tracts import save_ft, load_ft, nodes_by_index_mega, nodes_labels_mega, non_weighted_con_mat_mega, load_dwi_files, draw_con_mat
from FT.all_subj import all_subj_folders, all_subj_names
import numpy as np
from dipy.tracking.streamline import values_from_volume
import nibabel as nib
import os


def load_fibers(main_folder,n,index_to_text_file,fig_type,nii_file):
    tract_file = main_folder + r'\streamlines' + n + '_'+fig_type+'.trk'
    streamlines = load_ft(tract_file,nii_file)
    lab_labels_index, affine = nodes_by_index_mega(main_folder)
    labels_headers, idx = nodes_labels_mega(index_to_text_file)
    new_data, m, grouping = non_weighted_con_mat_mega(streamlines, lab_labels_index, affine, idx, main_folder)
    h = labels_headers
    return labels_headers, idx, m, grouping, h


def create_empty_vars():
    clean_grouping = {}
    m_clean = np.zeros(m.shape)
    clean_streamlines = []
    m_weighted = np.zeros(m.shape)

    return clean_grouping, m_clean, clean_streamlines, m_weighted


def find_bvec(main_folder):
    for file in os.listdir(main_folder):
        if file.endswith(".bvec"):
            bvec_file = os.path.join(main_folder, file)

    return bvec_file


def extract_weighted_data(bvec_file,weight_by):
    weight_by_file = bvec_file[:-5:] + '_' + weight_by + '.nii'
    weight_by_img = nib.load(weight_by_file)
    weight_by_data = weight_by_img.get_data()
    affine = weight_by_img.affine

    return weight_by_data, affine


def clean_non_cc(grouping, idx, clean_grouping, clean_streamlines):
    for pair, tracts in grouping.items():
        if pair[0] == 0 or pair[1] == 0:
            continue
        else:
            if pair[0] in list(np.asarray(idx[0:49])+1) and pair[1] in list(np.asarray(idx[50:99])+1):
                clean_grouping[pair] = tracts
                clean_streamlines+=tracts
            if pair[1] in list(np.asarray(idx[0:49])+1) and pair[0] in list(np.asarray(idx[50:99])+1):
                clean_grouping[pair] = tracts
                clean_streamlines+=tracts
    return clean_grouping, clean_streamlines


def cleaned_tracts_to_mat(clean_grouping, m_clean, weight_by_data, affine, m_weighted):
    for pair, tracts in clean_grouping.items():
        m_clean[pair[0]-1, pair[1]-1] = tracts.__len__()
        m_clean[pair[1]-1, pair[0]-1] = tracts.__len__()
        mean_vol_per_tract = []
        vol_per_tract = values_from_volume(weight_by_data, tracts, affine=affine)
        for s in vol_per_tract:
            mean_vol_per_tract.append(np.mean(s))
        mean_path_vol = np.nanmean(mean_vol_per_tract)
        m_weighted[pair[0] - 1, pair[1] - 1] = mean_path_vol
        m_weighted[pair[1] - 1, pair[0] - 1] = mean_path_vol


    return m_clean, m_weighted


if __name__ == '__main__':
    subj = all_subj_folders
    names = all_subj_names
    index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlascortex2nii.txt'
    fig_types = ['cc_1d','genu_1d','body_1d','splenium_1d']
    weight_by='1.5_2_AxPasi5'
    a= True
    for s, n in zip(subj, names):
        for fig_type in fig_types:
            main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s
            gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(main_folder)
            labels_headers, idx, m, grouping, h = load_fibers(main_folder, n, index_to_text_file,fig_type,nii_file)
            #if os.path.exists(os.path.join(main_folder, 'non-weighted('+fig_type+', MegaAtlas).png')):
            #if not os.path.exists(os.path.join(main_folder, 'non-weighted(' + fig_type + ', MegaAtlas).png')):
            if a:
                clean_grouping, m_clean, clean_streamlines, m_weighted = create_empty_vars()
                weight_by_data, affine = extract_weighted_data(bvec_file,weight_by)
                clean_grouping, clean_streamlines = clean_non_cc(grouping, idx, clean_grouping, clean_streamlines)
                m_clean, m_weighted = cleaned_tracts_to_mat(clean_grouping, m_clean, weight_by_data, affine, m_weighted)

                mm = m_clean[idx]
                mm = mm[:, idx]
                new_data = 1 / mm  # values distribute between 0 and 1, 1 represents distant nodes (only 1 tract)
                np.save(main_folder + r'\non-weighted_mega_'+fig_type+'_nonnorm_cleaned', mm)
                np.save(main_folder + r'\non-weighted_mega_'+fig_type+'_cleaned', new_data)

                save_ft(main_folder, n, clean_streamlines, nii_file, file_name='_'+ fig_type+'_cleaned.trk')
                fig_name = main_folder + r'\non-weighted('+fig_type+', MegaAtlas).png'
                draw_con_mat(new_data, h, fig_name, is_weighted=False)

                mm_weighted = m_weighted[idx]
                mm_weighted = mm_weighted[:, idx]
                new_data = 1 / (mm_weighted * 1.7 * 8.75)  # 1.7 - voxel resolution, 8.75 - axon diameter 2 ACV constant
                np.save(main_folder + r'\weighted_mega_'+fig_type+'_nonnorm_cleaned', mm_weighted)
                np.save(main_folder + r'\weighted_mega_'+fig_type+'_cleaned', new_data)

                fig_name = main_folder + r'\weighted('+fig_type+', MegaAtlas).png'
                draw_con_mat(new_data, h, fig_name, is_weighted=True)
