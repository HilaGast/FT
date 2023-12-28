import os, glob

subj_folders = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
old_names = ['data_3_2_AxPasi7.nii.gz','data_FA.nii.gz','data_MD.nii.gz']
new_names = ['ADD.nii.gz','FA.nii.gz','MD.nii.gz']

for subj_fol in subj_folders:
    for old_name,new_name in zip(old_names,new_names):
        old_name = subj_fol+old_name
        if os.path.exists(old_name):
            new_name = subj_fol+new_name
            try:
                os.rename(old_name,new_name)
            except FileExistsError: #delete the old file
                os.remove(old_name)

    print(subj_fol)
