import os, glob
from shutil import copy

stromboli_folders = glob.glob(f'Y:\hcp\*{os.sep}')
move_to_folder = r'F:\data\V7\HCP'
source_files = [r'AxCaliber\3_2_AxPasi7.nii.gz',r'AxCaliber\FA.nii.gz',r'AxCaliber\MD.nii.gz',r'tracts\HCP.tck',r'raw_data\mprage.nii.gz',r'raw_data\data.nii.gz',r'nodif\HCP.nii.gz',r'raw_data\brain_mask.nii.gz',r'raw_data\bvals',r'raw_data\bvecs']
dest_files = [f'data_3_2_AxPasi7.nii.gz',f'data_FA.nii.gz',f'data_MD.nii.gz',f'streamlines{os.sep}HCP_tracts.tck','MPRAGE.nii.gz','data.nii.gz','data_1st.nii.gz','hifi_nodif_brain_mask.nii.gz','data.bval','data.bvec']
# choose subjects:

for stro_fol in stromboli_folders:
    num = stro_fol.split(os.sep)[-2]
    dest_dir_name = f'{move_to_folder}{os.sep}{num}{os.sep}'
    if not os.path.exists(stro_fol+r'AxCaliber\3_2_AxPasi7.nii.gz'):
        continue
    if not os.path.exists(stro_fol+r'tracts\HCP.tck'):
        continue
    if not os.path.exists(dest_dir_name):
        os.mkdir(dest_dir_name)
    if not os.path.exists(dest_dir_name+f'streamlines{os.sep}'):
        os.mkdir(dest_dir_name+f'streamlines{os.sep}')
    if not os.path.exists(dest_dir_name+f'cm{os.sep}'):
        os.mkdir(dest_dir_name+f'cm{os.sep}')
    if os.path.exists(dest_dir_name+r'data_3_2_AxPasi7.nii.gz') and os.path.exists(dest_dir_name+r'streamlines\HCP_tracts.tck'):
        continue
    for sf,df in zip(source_files,dest_files):
        sfn = stro_fol+sf
        dfn = dest_dir_name+df
        try:
            copy(sfn,dfn)
        except FileNotFoundError:
            continue
