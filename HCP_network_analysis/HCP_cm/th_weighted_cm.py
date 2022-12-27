import glob, os
import numpy as np
from skimage.exposure import match_histograms

def th_hist_match(mat, mat_ref, th_val):
    matched = match_histograms(mat, mat_ref)
    matched[matched <= th_val] = 0

    return matched

def th_by_val(mat, val_min = 0):
    mat[mat<val_min] = 0

    return mat


if __name__ == '__main__':

    subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
    atlas = 'yeo7_200'
    for s in subj_list:
        cm_folder = s+f'cm{os.sep}'

        num_mat = np.load(cm_folder + f'{atlas}_Num_Org_SC_cm_ord.npy')
        add_mat = np.load(cm_folder+f'{atlas}_ADD_Org_SC_cm_ord.npy')
        #fa_mat = np.load(cm_folder + f'{atlas}_FA_Org_SC_cm_ord.npy')
        dist_mat = np.load(cm_folder + f'{atlas}_Dist_Org_SC_cm_ord.npy')

        num_th_histmatch = th_hist_match(num_mat, num_mat, 0)
        add_th_histmatch = th_hist_match(add_mat, num_mat, 0)
        dist_th_histmatch = th_hist_match(dist_mat, num_mat, 0)

        #num_add_histmatch = th_hist_match(num_th_histmatch*add_th_histmatch, num_mat, 1)
        num_dist_histmatch = th_hist_match(num_th_histmatch*dist_th_histmatch, num_mat, 1)
        add_dist_histmatch = th_hist_match(add_th_histmatch*dist_th_histmatch, num_mat, 1)


        num_th_histmatch = th_hist_match(num_mat, num_mat, 1)
        #add_th_histmatch = th_hist_match(add_mat, num_mat, 1)
        #fa_th_histmatch = th_hist_match(fa_mat, num_mat, 1)
        dist_th_histmatch = th_hist_match(dist_mat, num_mat, 1)

        #np.save(cm_folder+ f'{atlas}_Num_HistMatch_SC_cm_ord.npy', num_th_histmatch)
        #np.save(cm_folder + f'{atlas}_ADD_HistMatch_SC_cm_ord.npy', add_th_histmatch)
        #np.save(cm_folder + f'{atlas}_FA_HistMatch_SC_cm_ord.npy', fa_th_histmatch)
        np.save(cm_folder + f'{atlas}_Dist_HistMatch_SC_cm_ord.npy', dist_th_histmatch)
        #np.save(cm_folder + f'{atlas}_NumxADD_HistMatch_SC_cm_ord.npy', num_add_histmatch)
        np.save(cm_folder + f'{atlas}_NumxDist_HistMatch_SC_cm_ord.npy', num_dist_histmatch)
        np.save(cm_folder + f'{atlas}_ADDxDist_HistMatch_SC_cm_ord.npy', add_dist_histmatch)
