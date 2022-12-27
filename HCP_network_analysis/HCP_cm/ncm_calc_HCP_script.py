import glob, os
from HCP_network_analysis.HCP_cm.euclidean_distance_matrix import *
from HCP_network_analysis.HCP_cm.calc_network_communication_measures import *


def analyze_files(subj_list,atlas_name):
    for sl in subj_list:
        cm_files = glob.glob(f'{sl}cm{os.sep}{atlas_name}*SC_cm_ord.npy')

        #labels_file_path = find_labels_file(cm_files[0])
        #label_ctd = find_labels_centroids(labels_file_path)
        #euc_mat = euc_dist_mat(label_ctd,np.load(cm_files[0]))

        for cmf in cm_files:
            file_name = cmf.split(os.sep)[-1]
            cm = np.load(cmf)

            if 'HistMatch_SC' or '_Num_' in file_name:
                cm = np.log10(cm/np.nanmax(cm[:])+1)
            else:
                cm = 1/cm

            spe = calc_spe(cm)
            #ne = calc_ne(cm,euc_mat)

            spe_name = cmf.replace('SC','SPE')
            #ne_name = cmf.replace('SC','NE')

            np.save(spe_name, spe)
            #np.save(ne_name, ne)


if __name__ == '__main__':
    subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
    #analyze_files(subj_list, 'yeo7_200')
    analyze_files(subj_list, 'bna')







