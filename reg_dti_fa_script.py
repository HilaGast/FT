

import os
folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep\questionnaire'

all_subj_folders = os.listdir(folder_name)
subj = all_subj_folders

for s in subj:
    subj_name = r'/' + s + r'/'
    subj_folder = folder_name + subj_name
    subj_folder = subj_folder.replace(os.sep, '/')
    subj_folder = subj_folder.replace('C:', '/mnt/c')

    subj_first_charmed = subj_folder + 'diff_corrected_1st.nii'
    subj_fa = subj_folder + 'dti_fa.nii'
    out_registered = subj_folder + 'rdti_fa.nii'
    out_registered_mat = out_registered[:-4] + '.mat'
    options = '-bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'

    cmd = 'bash -lc "flirt -ref {0} -in {1} -out {2} -omat {3} {4}"'.format(subj_first_charmed, subj_fa, out_registered,
                                                                        out_registered_mat, options)
    cmd = cmd.replace(os.sep, '/')
    os.system(cmd)
