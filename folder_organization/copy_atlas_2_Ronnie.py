import os, glob
from shutil import copy

my_hcp_folder = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
for fol in my_hcp_folder:
    subj_num = str.split(fol, os.sep)[-2]
    atlas_file_name = f'{fol}ryeo7_200_atlas.nii'
    if os.path.exists(atlas_file_name):
        new_atlas_file_name = f'Y:\qnap_hcp\{subj_num}{os.sep}atlas{os.sep}yeo7_200_atlas.nii'
        copy(atlas_file_name, new_atlas_file_name)