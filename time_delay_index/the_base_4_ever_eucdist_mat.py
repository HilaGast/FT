import os.path

import numpy as np

from HCP_network_analysis.HCP_cm.euclidean_distance_matrix import *
import glob

if __name__ == '__main__':

    main_fol = 'F:\Hila\TDI\TheBase4Ever'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}*[0-9]{os.sep}')
    all_atlas = ['yeo7_100', 'yeo7_200','bnacor']
    cm_name_extras = ''

    for atlas in all_atlas:
        for subj_fol in all_subj_fol:
            cm_name = f'{subj_fol}cm{os.sep}{atlas}_Num{cm_name_extras}_cm_ord.npy'
            euc_dist_cm_name = f'{subj_fol}cm{os.sep}{atlas}_EucDist{cm_name_extras}_cm_ord.npy'
            if not os.path.exists(euc_dist_cm_name) and os.path.exists(cm_name):
                cm = np.load(cm_name)
                labels_file_path = find_labels_file(cm_name)
                label_ctd = find_labels_centroids(labels_file_path)
                euc_mat = euc_dist_mat(label_ctd, cm)
                np.save(euc_dist_cm_name, euc_mat)



