import numpy as np
import nibabel as nib
import os
from Tractography.fiber_weighting import weight_streamlines_by_cm
from Tractography.files_loading import load_ft
from Tractography.tractography_vis import show_tracts_simple

# choose representative subject:
save_to_folder = 'F:\Hila\TDI\siemens\group_cm'
#subj_fol = r'F:\Hila\TDI\siemens\C1156_05\D60d11'
#dat_file = subj_fol+os.sep+'diff_corrected_D60d11.nii'
subj_fol = r'G:\data\V7\HCP\100307'
dat_file = subj_fol+os.sep+'data.nii.gz'

affine = nib.load(dat_file).affine
#tract_name = subj_fol+os.sep+'streamlines'+os.sep+'wb_csd_fa.tck'
tract_name = subj_fol+os.sep+'streamlines'+os.sep+'HCP_tracts.tck'
streamlines = load_ft(tract_name, dat_file)
lab = subj_fol+r'\rnewBNA_Labels.nii'

# choose cm for weighting:
cm = np.load(r'F:\Hila\TDI\siemens\group_cm\median_time_th3_bnacor_D60d11_h.npy')
cm[np.isnan(cm)] = 0
cm_lookup = np.load(r'F:\Hila\TDI\siemens\group_cm\bnacor_cm_ord_lookup.npy')

# choose labels for weighting:
lab_labels = nib.load(lab).get_fdata()
lab_labels_index = [labels for labels in lab_labels]
labels = np.asarray(lab_labels_index, dtype='int')

# weight streamlines:
s_list, vec_vols = weight_streamlines_by_cm(streamlines, affine, labels, cm, cm_lookup)

# show weighted streamlines:

#show_tracts_simple(s_list, save_to_folder, 'median_ms', time2present=4, down_samp=2, vec_vols=vec_vols, colormap = 'seismic',min=-7,max=7, weighted=True)
show_tracts_simple(s_list, save_to_folder, 'median_h', time2present=4, down_samp=2, vec_vols=vec_vols, scale = [0,600], weighted=True)