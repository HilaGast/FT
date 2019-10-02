import numpy as np
from FT.all_subj import all_subj_names
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from FT.weighted_tracts import nodes_labels_mega

if __name__ == '__main__':
    subj = all_subj_names
    index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlascortex2nii.txt'
    labels_headers, idx = nodes_labels_mega(index_to_text_file)
    h = labels_headers
    cc = np.zeros((len(h),len(h),len(subj)))
    genu = np.zeros((len(h),len(h),len(subj)))
    splenium = np.zeros((len(h),len(h),len(subj)))
    body = np.zeros((len(h),len(h),len(subj)))
    for i, s in enumerate(subj):
        main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5' + s

        # load cc:
        fig_type = 'cc_cortex_clean'
        cc[:,:,i] = np.load(main_folder + r'\weighted_mega_'+fig_type+ '.npy')
        #load genu:
        fig_type = 'genu_cortex_clean'
        genu[:,:,i] = np.load(main_folder + r'\weighted_mega_'+fig_type+ '.npy')
        # load splenium:
        fig_type = 'splenium_cortex_clean'
        splenium[:,:,i] = np.load(main_folder + r'\weighted_mega_'+fig_type+ '_nonnorm.npy')
        # load body:
        fig_type = 'body_before_clean'
        body[:,:,i] = np.load(main_folder + r'\weighted_mega_'+fig_type+ '_nonnorm.npy')

    # calculate average values
    cc_n = cc
    #cc_n[cc_n>1] = np.nan
    m_cc= np.nanmean(cc_n,axis=2)
    genu_n = genu
    #genu_n[genu_n>1] = np.nan
    m_genu = np.nanmean(genu_n,axis=2)
    splenium_n = splenium
    #splenium_n[splenium_n>1] = np.nan
    m_splenium = np.nanmean(splenium_n,axis=2)
    body_n = body
    #body_n[body_n>1] = np.nan
    m_body = np.nanmean(body_n,axis=2)


    mats = [m_cc, m_genu, m_body, m_splenium]
    names = ['\mean_cc_nonnorm','\mean_genu_nonnorm','\mean_body_nonnorm','\mean_splenium_nonnorm']
    # display on con mat
    mat_title = 'Weighted connectivity matrix (average values)'
    for mat,name in zip(mats,names):
        fig_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5'+name
        #mat[~np.isfinite(mat)] = 2
        plt.figure(1, [24, 20])
        cmap = cm.YlOrRd
        cmap.set_over('black')
        plt.imshow(mat, interpolation='nearest', cmap=cmap, origin='upper', vmax=1)
        plt.colorbar()
        plt.xticks(ticks=np.arange(0, len(mat), 1), labels=h)
        plt.yticks(ticks=np.arange(0, len(mat), 1), labels=h)
        plt.title(mat_title, fontsize=20)
        plt.tick_params(axis='x', pad=8.0, labelrotation=90, labelsize=9)
        plt.tick_params(axis='y', pad=8.0, labelsize=9)
        plt.savefig(fig_name+'.png')
        np.save(fig_name,mat)
        plt.show()

