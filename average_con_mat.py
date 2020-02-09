import numpy as np
from FT.all_subj import all_subj_names, all_subj_folders
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from FT.weighted_tracts import *
import matplotlib.colors as colors


if __name__ == '__main__':
    subj = all_subj_folders
    index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlas2nii.txt'
    labels_headers, idx = nodes_labels_mega(index_to_text_file)
    h = labels_headers
    cc = np.zeros((len(h),len(h),len(subj)))
    genu = np.zeros((len(h),len(h),len(subj)))
    splenium = np.zeros((len(h),len(h),len(subj)))
    body = np.zeros((len(h),len(h),len(subj)))
    w = np.zeros((len(h),len(h),len(subj)))
    nw = np.zeros((len(h),len(h),len(subj)))

    for i, s in enumerate(subj):
        main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s

        #load wholebrain:
        fig_type = 'wholebrain_plus_new2_nonnorm'
        w[:,:,i] = np.load(main_folder + r'\weighted_mega_'+fig_type+ '.npy')

        fig_type = 'wholebrain_plus_new2_nonnorm'
        nw[:,:,i] = np.load(main_folder + r'\non-weighted_mega_'+fig_type+ '.npy')

    '''calculate average&std values'''
    w_n = w.copy()
    m_w = np.zeros((len(h),len(h)))
    s_w = np.zeros((len(h),len(h)))
    for r in range(len(h)):
        for c in range(len(h)):
            w_n_rc = w_n[r,c,:]
            if np.count_nonzero(w_n_rc)<len(subj)/5:
                m_w[r, c] = 0
                s_w[r, c] = 0
            else:
                m_w[r,c]= np.nanmean(w_n_rc[w_n_rc>0])
                s_w[r,c] = np.nanstd(w_n_rc[w_n_rc>0])

    nw_n = nw.copy()
    m_nw = np.zeros((len(h),len(h)))
    s_nw = np.zeros((len(h),len(h)))
    for r in range(len(h)):
        for c in range(len(h)):
            nw_n_rc = nw_n[r,c,:]
            if np.count_nonzero(nw_n_rc)<len(subj)/5:
                m_nw[r, c] = 0
                s_nw[r, c] = 0
            else:
                m_nw[r,c]= np.nanmean(nw_n_rc[nw_n_rc>0])
                s_nw[r,c] = np.nanstd(nw_n_rc[nw_n_rc>0])
    '''calculate average values'''
    m_nw_norm = 1/m_nw
    m_w_norm = 1/(m_w*8.75)

    mats = [m_w, m_nw, s_w, s_nw, m_w_norm, m_nw_norm]
    #names = ['\mean_cc_nonnorm','\mean_genu_nonnorm','\mean_body_nonnorm','\mean_splenium_nonnorm']
    names = ['\mean_w_plus','\mean_nw_plus','\std_w_plus','\std_nw_plus','\mean_w_plus_norm','\mean_nw_plus_norm']
    weight = [True, False,True, False,True, False]
    # display on con mat
    for mat,name,isw in zip(mats,names,weight):
        #norm:
        fig_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + name
        #non-norm:
        fig_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'+name
        np.save(fig_name,mat)
        #mat[mat>1] = np.nan
        draw_con_mat(mat, h, fig_name, is_weighted=isw)

