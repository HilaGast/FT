import glob,os

from fsl.file_prep import *

subj, folder_name, atlas_template, atlas_label = basic_files(False, atlas_type='bna_cor')
main_folder = r'F:\data\V7\TheBase4Ever'

#for subj_fol in glob.glob(main_folder + f'{os.sep}e*{os.sep}*{os.sep}*'):
for subj_fol in glob.glob(main_folder + f'{os.sep}*{os.sep}'):
    subj_name = subj_fol + os.sep
    subj_folder = subj_name.replace(os.sep, '/')

    mprage_file_name, diff_file_name, pa_file_name = subj_files(subj_folder)
    out_registered = subj_folder + 'r' + mprage_file_name[:-4]+'_brain.nii'

    warp_name = subj_folder + 'atlas2subj.nii'

    subj_folder = os_path_2_fsl(subj_folder)
    atlas_label = os_path_2_fsl(atlas_label)
    out_registered = os_path_2_fsl(out_registered)
    warp_name = os_path_2_fsl(warp_name)

    atlas_labels_registered = apply_fnirt_warp_on_label(subj_folder, atlas_label, out_registered, warp_name)

