import glob,os

from fsl.file_prep import os_path_2_fsl, subj_files, apply_fnirt_warp_on_label, basic_files

main_folder = r'F:\Hila\balance'
subj, folder_name, atlas_template, atlas_label = basic_files(False, atlas_type='yeo7_200', folder_name=main_folder)

#for subj_fol in glob.glob(main_folder + f'{os.sep}e*{os.sep}*{os.sep}*'):
for subj_fol in glob.glob(main_folder + f'{os.sep}e*{os.sep}before{os.sep}*{os.sep}'):
    subj_folder = subj_fol.replace(os.sep, '/')

    mprage_file_name, diff_file_name, pa_file_name = subj_files(subj_folder)
    out_registered = subj_folder + 'r' + mprage_file_name[:-4]+'_brain.nii'

    warp_name = subj_folder + 'atlas2subj.nii'

    subj_folder = os_path_2_fsl(subj_folder)
    atlas_label = os_path_2_fsl(atlas_label)
    out_registered = os_path_2_fsl(out_registered)
    warp_name = os_path_2_fsl(warp_name)

    atlas_labels_registered = apply_fnirt_warp_on_label(subj_folder, atlas_label, out_registered, warp_name)

