import nibabel as nib
from fsl.hcp_atlas_registration import basic_files_hcp
import os
import numpy as np


subj = basic_files_hcp(False)[0]
averaged_map = []
s = subj[0]
affine = nib.load(s + 'raverage_add_map.nii').affine
header = nib.load(s + 'raverage_add_map.nii').header
i=0
for s in subj[::]:
    if not os.path.exists(s + 'raverage_add_map.nii'):
        continue
    else:
        subj_map = nib.load(s + 'raverage_add_map.nii').get_fdata(dtype=np.float16)
        averaged_map.append(subj_map)
        i+=1

averaged_map = np.asarray(averaged_map)
averaged_map[averaged_map<0.3] = np.nan
averaged_map_mean = np.nanmean(averaged_map, axis=0)
wm_mask = nib.load(r'G:\data\atlases\BNA\tissues\MNI152_T1_1mm_brain_seg_2.nii').get_fdata()
averaged_map_masked = averaged_map_mean*wm_mask
map_img = nib.Nifti1Image(averaged_map_masked,affine, header)
nib.save(map_img,rf'G:\data\V7\HCP\avaraged_add_map_masked_{str(i)}_subjects_0.3th.nii')