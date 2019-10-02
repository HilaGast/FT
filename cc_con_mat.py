from FT.weighted_tracts import save_ft, load_ft, nodes_by_index_mega, nodes_labels_mega, non_weighted_con_mat_mega, load_dwi_files, draw_con_mat
from FT.all_subj import all_subj_names
import numpy as np
from dipy.tracking.streamline import values_from_volume
import nibabel as nib
import os


def load_fibers(s,index_to_text_file,fig_type):
    main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5' + s
    tract_file = main_folder + r'\streamlines' + s + '_'+fig_type+'.trk'
    streamlines = load_ft(tract_file)
    lab_labels_index, affine = nodes_by_index_mega(main_folder)
    labels_headers, idx = nodes_labels_mega(index_to_text_file)
    new_data, m, grouping = non_weighted_con_mat_mega(streamlines, lab_labels_index, affine, idx, main_folder)
    h = labels_headers
    return main_folder, labels_headers, idx, m, grouping, h


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
        if file.endswith("brain_seg.nii"):
            labels_file_name = os.path.join(main_folder, file)

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
    subj = all_subj_names[0:21]

    index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlascortex2nii.txt'
    fig_types = ['cc_cortex','genu_cortex','body_cortex','splenium_cortex']
    weight_by='pasiS'
    for s in subj:
        for fig_type in fig_types:
            main_folder, labels_headers, idx, m, grouping, h = load_fibers(s, index_to_text_file,fig_type)
            clean_grouping, m_clean, clean_streamlines, m_weighted = create_empty_vars()
            gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(main_folder)
            weight_by_data, affine = extract_weighted_data(bvec_file,weight_by)
            clean_grouping, clean_streamlines = clean_non_cc(grouping, idx, clean_grouping, clean_streamlines)
            m_clean, m_weighted = cleaned_tracts_to_mat(clean_grouping, m_clean, weight_by_data, affine, m_weighted)

            mm = m_clean[idx]
            mm = mm[:, idx]
            new_data = 1 / mm  # values distribute between 0 and 1, 1 represents distant nodes (only 1 tract)
            new_data[new_data > 1] = 2
            np.save(main_folder + r'\non-weighted_mega_'+fig_type+'_cleaned', new_data)

            save_ft(main_folder, clean_streamlines, file_name='_'+ fig_type+'_cleaned.trk')
            fig_name = main_folder + r'\non-weighted('+fig_type+', MegaAtlas).png'
            draw_con_mat(new_data, h, fig_name, is_weighted=False)

            mm_weighted = m_weighted[idx]
            mm_weighted = mm_weighted[:, idx]
            mm_weighted[mm_weighted<0.01] = 0
            new_data = (10-mm_weighted)/10 # normalization between 0 and 1
            new_data[new_data ==1] = 2
            np.save(main_folder + r'\weighted_mega_'+fig_type+'_cleaned', new_data)

            fig_name = main_folder + r'\weighted('+fig_type+', MegaAtlas).png'
            draw_con_mat(new_data, h, fig_name, is_weighted=True)
