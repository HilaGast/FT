import numpy as np
from os.path import join as pjoin
from weighted_tracts import *


def calc_avg_mat(subj,fig_type, calc_type='mean', draw = True, isw = True):
    #labels_headers, idx = nodes_labels_aal3(index_to_text_file)
    labels_headers, idx = nodes_labels_yeo7(index_to_text_file)
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
    #cond = 'ec_before'
    fig_name_m = pjoin(r'F:\Hila\Ax3D_Pack\mean_vals\yeo7_200',f'{calc_type}_{fig_type}')
    #fig_name_m = pjoin(r'F:\Hila\balance','mean_mat',f'{calc_type}_{fig_type}_{cond}')
    np.save(fig_name_m,mat_m)
    #fig_name_s = pjoin(r'F:\Hila\Ax3D_Pack\mean_vals\aal3_atlas',f'std_{fig_type}')
    #np.save(fig_name_s,mat_s)
'''
    if isw:
        mat_m_norm = 1/(mat_m*8.75)
        np.save(fig_name_m,mat_m_norm)

    else:
        mat_m_norm = 1/mat_m
        np.save(fig_name_m,mat_m_norm)

    if draw:
        draw_con_mat(mat_m_norm, h, fig_name_m, is_weighted=isw)
'''

if __name__ == '__main__':
    subj = all_subj_folders
    fig_type = 'non-weighted_wholebrain_4d_labmask_yeo7_200_nonnorm'
    mean_mat = calc_avg_mat(subj,fig_type,calc_type='mean', draw = True, isw = True)
    #median_mat = calc_avg_mat(subj,fig_type,calc_type='median', draw = True)
