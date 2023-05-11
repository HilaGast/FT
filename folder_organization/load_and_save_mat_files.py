import numpy as np
from weighted_tracts import nodes_labels_yeo7
import scipy.io as sio
import os, glob


subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
th = 'Org'
atlas = 'yeo7_200'

for sl in subj_list:

    add_cm = np.load(f'{sl}cm{os.sep}{atlas}_ADD_{th}_SC_cm_ord.npy')
    num_cm = np.load(f'{sl}cm{os.sep}{atlas}_Num_{th}_SC_cm_ord.npy')
    fa_cm = np.load(f'{sl}cm{os.sep}{atlas}_FA_{th}_SC_cm_ord.npy')


    mat_file_name = rf'{sl}{os.sep}cm{os.sep}{atlas}_{th}.mat'
    sio.savemat(mat_file_name, {'axsi': add_cm,'number_of_tracts':num_cm,'fa':fa_cm})
