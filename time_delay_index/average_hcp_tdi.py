import glob

import numpy as np
import os
from os.path import join as pjoin

def calc_avg_mat(subj,fig_type,fol_2_save, calc_type='mean', atlas_type = 'yeo7_200', adds_for_file_name='',adds_for_file_save=''):

    idxl = len(np.load(pjoin(subj[0],'cm',f'{atlas_type}_cm_ord_lookup.npy')))
    all_subj_mat = np.zeros((idxl, idxl, len(subj)))
    for i, s in enumerate(subj):
        all_subj_mat[:,:,i] = np.load(f'{s}{os.sep}cm{os.sep}{atlas_type}_{fig_type}_{adds_for_file_name}cm_ord.npy')
    all_subj_mat[all_subj_mat == 0] = np.nan
    '''calculate average&std values'''
    subj_mat = all_subj_mat.copy()
    mat_m = np.zeros((idxl,idxl))
    mat_s = np.zeros((idxl,idxl))
    for r in range(idxl):
        for c in range(idxl):
            rc = subj_mat[r,c,:]
            if np.count_nonzero(~np.isnan(rc)) < len(subj)/5:
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
    if len(adds_for_file_name) > 0:
        adds_for_file_name = '_' + adds_for_file_name
    fig_name_m = pjoin(fol_2_save,f'{calc_type}_{atlas_type}_{fig_type}{adds_for_file_name}{adds_for_file_save}')
    #fig_name_m = pjoin(r'F:\Hila\balance','mean_mat',f'{calc_type}_{fig_type}_{cond}')
    np.save(fig_name_m,mat_m)
    #fig_name_s = pjoin(r'F:\Hila\Ax3D_Pack\mean_vals\aal3_atlas',f'std_{fig_type}')
    #np.save(fig_name_s,mat_s)


if __name__ == '__main__':
    main_fol = 'G:\data\V7\HCP'
    atlas = 'yeo7_100'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}*[0-9]{os.sep}')
    weights = ['Num', 'ADD', 'Dist', 'FA', 'time_th3','fmri']
    for w in weights:
        subjects=[]
        for s in all_subj_fol:
            file_name = f'{s}cm{os.sep}{atlas}_{w}_Org_SC_cm_ord.npy'
            if os.path.exists(file_name):
                subjects.append(s)
        calc_avg_mat(subjects, w, main_fol + os.sep + 'cm', 'median', atlas, 'Org_SC')