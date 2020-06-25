import numpy as np
from os.path import join as pjoin
from weighted_tracts import *


def calc_avg_mat(subj,fig_type, calc_type='mean', draw = True, isw = True):
    index_to_text_file = r'C:\Users\hila\data\megaatlas\megaatlas2nii.txt'
    #index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlas2nii.txt'
    labels_headers, idx = nodes_labels_mega(index_to_text_file)
    h = labels_headers
    all_subj_mat = np.zeros((len(h),len(h),len(subj)))
    for i, s in enumerate(subj):
        main_folder = subj_folder + s
        all_subj_mat[:,:,i] = np.load(f'{main_folder}\{fig_type}.npy')

    '''calculate average&std values'''
    subj_mat = all_subj_mat.copy()
    mat_m = np.zeros((len(h),len(h)))
    mat_s = np.zeros((len(h),len(h)))
    for r in range(len(h)):
        for c in range(len(h)):
            rc = subj_mat[r,c,:]
            if np.count_nonzero(rc) < len(subj)/5:
                mat_m[r, c] = 0
                mat_s[r, c] = 0
            else:
                if calc_type == 'mean':
                    mat_m[r,c]= np.nanmean(rc[rc>0])
                    mat_s[r,c] = np.nanstd(rc[rc>0])
                elif calc_type == 'median':
                    mat_m[r, c] = np.nanmedian(rc[rc > 0])
                    mat_s[r, c] = np.nanstd(rc[rc > 0])
    '''calculate average values'''
    fig_name_m = pjoin(r'C:\Users\hila\data\mats',f'{calc_type}_{fig_type}')
    fig_name_s = pjoin(r'C:\Users\hila\data\mats',f'std_{fig_type}')
    np.save(fig_name_s,mat_s)

    if isw:
        mat_m_norm = 1/(mat_m*8.75)
        np.save(fig_name_m,mat_m_norm)

    else:
        mat_m_norm = 1/mat_m
        np.save(fig_name_m,mat_m_norm)

    if draw:
        draw_con_mat(mat_m_norm, h, fig_name_m, is_weighted=isw)


if __name__ == '__main__':
    subj = all_subj_folders
    fig_type = 'weighted_mega_SLF_nonnorm'
    mean_mat = calc_avg_mat(subj,fig_type,calc_type='mean', draw = True, isw = True)
    #median_mat = calc_avg_mat(subj,fig_type,calc_type='median', draw = True)
