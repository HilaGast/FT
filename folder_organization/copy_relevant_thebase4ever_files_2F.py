import glob, os

def find_folders(main_folder, folder_name):
    return glob.glob(f'{main_folder}{os.sep}*{os.sep}')

def create_folders(main_folder, sub_folders_list):
    dest_sub_list = []
    for folder in sub_folders_list:
        file_folder = folder.split(os.sep)[-2]
        file_num = file_folder.split('_')[-3]
        folder_name = rf'{main_folder}{os.sep}{file_num}'
        dest_sub_list.append(folder_name)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        if not os.path.exists(folder_name + os.sep + 'streamlines'):
            os.mkdir(folder_name + os.sep + 'streamlines')
        if not os.path.exists(folder_name + os.sep + 'cm'):
            os.mkdir(folder_name + os.sep + 'cm')
    return dest_sub_list

def copy_files(sub_folders_list, dest_sub_list):
    from shutil import copy

    for org_fol, dest_fol in zip(sub_folders_list[38:], dest_sub_list[38:]):
        subj_num = dest_fol.split(os.sep)[-1]
        # copy streamlines:
        org_file = glob.glob(f'{org_fol}streamlines{os.sep}*_wholebrain_5d_labmask_msmt.trk')[0]
        copy(org_file, dest_fol + os.sep + 'streamlines' + os.sep + f'tracts.trk')
        # copy MPRAGE:
        org_file = glob.glob(f'{org_fol}r*MPRAGERL*_brain.nii')[0]
        copy(org_file, dest_fol + os.sep + f'rMPRAGE_brain.nii')
        org_file = glob.glob(f'{org_fol}[0-9]*MPRAGERL*_brain.nii')[0]
        copy(org_file, dest_fol + os.sep + f'MPRAGE_brain.nii')
        # copy diffusion data:
        copy(f'{org_fol}diff_corrected.nii', dest_fol + os.sep + f'diff_corrected.nii')
        copy(f'{org_fol}diff_corrected_1st.nii', dest_fol + os.sep + f'diff_corrected_1st.nii')
        copy(f'{org_fol}diff_corrected.bval', dest_fol + os.sep + f'diff_corrected.bval')
        copy(f'{org_fol}diff_corrected.bvec', dest_fol + os.sep + f'diff_corrected.bvec')
        # copy AxSI file:
        copy(f'{org_fol}diff_corrected_3_2_AxPasi7.nii', dest_fol + os.sep + f'ADD.nii')
        # copy registered atlas:
        copy(f'{org_fol}rnewBNA_Labels.nii', dest_fol + os.sep + f'rnewBNA_Labels.nii')
        copy(f'{org_fol}ryeo7_200_atlas.nii', dest_fol + os.sep + f'ryeo7_200_atlas.nii')
        copy(f'{org_fol}ryeo7_1000_atlas.nii', dest_fol + os.sep + f'ryeo7_1000_atlas.nii')
        copy(f'{org_fol}ryeo17_1000_atlas.nii', dest_fol + os.sep + f'ryeo7_1000_atlas.nii')





