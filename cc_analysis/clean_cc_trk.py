from weighted_tracts import save_ft, load_ft, nodes_by_index_general, nodes_labels_yeo7, non_weighted_con_mat_mega, load_dwi_files, draw_con_mat
from all_subj import all_subj_folders, all_subj_names,index_to_text_file,subj_folder
import numpy as np
import nibabel as nib
from remove_cci_outliers import remove_cci_outliers


def load_fibers(main_folder,n,index_to_text_file,fig_type,nii_file):
    tract_file = main_folder + r'\streamlines' + n + '_'+fig_type+'.trk'
    streamlines = load_ft(tract_file,nii_file)
    lab_labels_index, affine = nodes_by_index_general(main_folder, atlas='yeo7_200')
    labels_headers, idx = nodes_labels_yeo7(index_to_text_file)
    new_data, m, grouping = non_weighted_con_mat_mega(streamlines, lab_labels_index, affine, idx, main_folder)
    h = labels_headers
    return labels_headers, idx, m, grouping, h


def create_empty_vars():
    clean_grouping = {}
    m_clean = np.zeros(m.shape)
    clean_streamlines = []
    m_weighted = np.zeros(m.shape)

    return clean_grouping, m_clean, clean_streamlines, m_weighted


def extract_weighted_data(bvec_file,weight_by):
    weight_by_file = bvec_file[:-5:] + '_' + weight_by + '.nii'
    weight_by_img = nib.load(weight_by_file)
    weight_by_data = weight_by_img.get_fdata()
    affine = weight_by_img.affine

    return weight_by_data, affine


def clean_non_cc(grouping, idx, clean_grouping, clean_streamlines):
    for pair, tracts in grouping.items():
        if pair[0] == 0 or pair[1] == 0:
            continue
        else:
            if pair[0] in list(np.asarray(idx[0:int(len(idx)/2)])+1) and pair[1] in list(np.asarray(idx[int(len(idx)/2):len(idx)])+1):
                clean_streamlines+=tracts
            if pair[1] in list(np.asarray(idx[0:int(len(idx)/2)])+1) and pair[0] in list(np.asarray(idx[int(len(idx)/2):len(idx)])+1):
                clean_streamlines+=tracts

    clean_streamlines, keep_streamlines_idx = remove_cci_outliers(clean_streamlines)

    return clean_streamlines


if __name__ == '__main__':
    subj = all_subj_folders
    names = all_subj_names
    fig_type = 'cc_10d_labmask'
    weight_by='2_2_AxPasi7'

    for s, n in zip(subj[26::], names[26::]):
        main_folder = subj_folder + s
        gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(main_folder)
        labels_headers, idx, m, grouping, h = load_fibers(main_folder, n, index_to_text_file,fig_type,nii_file)
        clean_grouping, m_clean, clean_streamlines, m_weighted = create_empty_vars()
        weight_by_data, affine = extract_weighted_data(bvec_file,weight_by)
        clean_streamlines = clean_non_cc(grouping, idx, clean_grouping, clean_streamlines)
        save_ft(main_folder, n, clean_streamlines, nii_file, file_name='_'+ fig_type+'_cleaned.trk')

