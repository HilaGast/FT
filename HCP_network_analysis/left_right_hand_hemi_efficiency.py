import numpy as np
import glob, os
from network_analysis.global_network_properties import get_efficiency
import matplotlib.pyplot as plt

def find_handedness(subjects):
    import pandas as pd
    table1 = pd.read_csv('G:\data\V7\HCP\HCP_demographic_data.csv')
    handedness_index = []
    for s in subjects:
        subj_id = s.split('\\')[-3]
        handedness = float(table1['Handedness'][table1['Subject']==int(subj_id)].values)
        handedness_index.append(handedness)

    return handedness_index






if __name__ == '__main__':
    from HCP_network_analysis.hcp_cm_parameters import *

    atlas_index_labels = r'G:\data\atlases\yeo\yeo7_200\index2label.txt'
    ncm = ncm_options[0]
    atlas = atlases[0]
    weight_by = weights[0]
    regularization = reg_options[1]
    idx = np.load(rf'G:\data\V7\HCP\{atlas}_cm_ord_lookup.npy')
    mni_atlas_file_name = r'G:\data\atlases\yeo\yeo7_200\yeo7_200_atlas.nii'
    nii_base = r'G:\data\atlases\yeo\yeo7_200\Schaefer_template_brain.nii'
    output_folder = r'G:\data\V7\HCP\Eglob_by_hemi_and_handedness'
    subjects = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{weight_by}_{regularization}_{ncm}_cm_ord.npy')
    handedness_index = find_handedness(subjects)
    eglob_l = []
    eglob_r = []
    for s in subjects:
        cm = np.load(s)
        if atlas == 'yeo7_200':
            cm_l = cm[0:100, 0:100]
            cm_r = cm[100:200, 100:200]

        eglob_l.append(get_efficiency(cm_l))
        eglob_r.append(get_efficiency(cm_r))


    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Handedness (L-R)')
    ax1.set_ylabel('Eglob')
    ax1.scatter(handedness_index, eglob_l, color=[0.2, 0.7, 0.6])
    ax1.scatter(handedness_index, eglob_r, color=[0.2, 0.5, 0.8])
    fig.legend(['Left Hemisphere', 'Right Hemisphere'], loc=4)
    plt.ylim(0,0.15)
    plt.title(weight_by)
    fig.tight_layout()
    plt.show()

