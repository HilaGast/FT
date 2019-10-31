import numpy as np
from FT.all_subj import all_subj_names, all_subj_folders
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from FT.weighted_tracts import *
import matplotlib.colors as colors


if __name__ == '__main__':
    subj = all_subj_folders
    index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlascortex2nii.txt'
    labels_headers, idx = nodes_labels_mega(index_to_text_file)
    h = labels_headers
    cc = np.zeros((len(h),len(h),len(subj)))
    genu = np.zeros((len(h),len(h),len(subj)))
    splenium = np.zeros((len(h),len(h),len(subj)))
    body = np.zeros((len(h),len(h),len(subj)))
    w = np.zeros((len(h),len(h),len(subj)))

    for i, s in enumerate(subj):
        main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s

        #load wholebrain:
        fig_type = 'wholebrain_cortex_nonnorm'
        w[:,:,i] = np.load(main_folder + r'\non-weighted_mega_'+fig_type+ '.npy')

        # load cc:
        fig_type = 'cc_cortex_nonnorm_cleaned'
        #fig_type = 'cc_cortex_cleaned'

        cc[:,:,i] = np.load(main_folder + r'\non-weighted_mega_'+fig_type+ '.npy')
        #load genu:
        fig_type = 'genu_cortex_nonnorm_cleaned'
        #fig_type = 'genu_cortex_cleaned'
        genu[:,:,i] = np.load(main_folder + r'\non-weighted_mega_'+fig_type+ '.npy')
        # load splenium:
        fig_type = 'splenium_cortex_nonnorm_cleaned'
        #fig_type = 'splenium_cortex_cleaned'

        splenium[:,:,i] = np.load(main_folder + r'\non-weighted_mega_'+fig_type+ '.npy')
        # load body:
        fig_type = 'body_cortex_nonnorm_cleaned'
        #fig_type = 'body_cortex_cleaned'

        body[:,:,i] = np.load(main_folder + r'\non-weighted_mega_'+fig_type+ '.npy')

    # calculate average values
    w_n = w.copy()
    #w_n[w_n>1] = np.nan
    count_nan_mat = np.isfinite(w_n)
    bin_nan_mat = np.count_nonzero(count_nan_mat, axis=2)<(len(subj)/5)
    m_w= np.nanmedian(w_n,axis=2)
    #m_w[bin_nan_mat]= np.nan


    cc_n = cc.copy()
    #cc_n[cc_n>1] = np.nan
    count_nan_mat = np.isfinite(cc_n)
    bin_nan_mat = np.count_nonzero(count_nan_mat, axis=2)<(len(subj)/5)
    m_cc= np.nanmedian(cc_n,axis=2)
    #m_cc[bin_nan_mat]= np.nan

    genu_n = genu.copy()
    #genu_n[genu_n>1] = np.nan
    count_nan_mat = np.isfinite(genu_n)
    bin_nan_mat = np.count_nonzero(count_nan_mat, axis=2)<(len(subj)/5)
    m_genu = np.nanmedian(genu_n,axis=2)
    #m_genu[bin_nan_mat]= np.nan


    splenium_n = splenium.copy()
    #splenium_n[splenium_n>1] = np.nan
    count_nan_mat = np.isfinite(splenium_n)
    bin_nan_mat = np.count_nonzero(count_nan_mat, axis=2)<(len(subj)/5)
    m_splenium = np.nanmedian(splenium_n,axis=2)
    #m_splenium[bin_nan_mat]= np.nan

    body_n = body
    #body_n[body_n>1] = np.nan
    count_nan_mat = np.isfinite(body_n)
    bin_nan_mat = np.count_nonzero(count_nan_mat, axis=2)<(len(subj)/5)
    m_body = np.nanmedian(body_n,axis=2)
    #m_body[bin_nan_mat]= np.nan



    '''
    cc_n = cc
    m_cc = np.nanmean(cc_n, axis=2)
    genu_n = genu
    m_genu = np.nanmean(genu_n, axis=2)
    splenium_n = splenium
    m_splenium = np.nanmean(splenium_n, axis=2)
    body_n = body
    m_body = np.nanmean(body_n, axis=2)
    '''

    mats = [m_w, m_cc, m_genu, m_body, m_splenium]
    #names = ['\mean_cc_nonnorm','\mean_genu_nonnorm','\mean_body_nonnorm','\mean_splenium_nonnorm']
    names = ['\mean_w','\mean_cc','\mean_genu','\mean_body','\mean_splenium']

    # display on con mat
    mat_title = 'non-weighted connectivity matrix (average values)'
    for mat,name in zip(mats,names):
        fig_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'+name+'_num_nonnorm'
        np.save(fig_name,mat)
        draw_con_mat(mat, h, fig_name, is_weighted=False)

