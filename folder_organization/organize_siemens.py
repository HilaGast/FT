import glob, os, shutil

# main_folder = r'F:\Hila\TDI\siemens'
# subj_folders = glob.glob(f'{main_folder}{os.sep}[C,T]*{os.sep}')
# sub_folders = ['D60d11', 'D45d13', 'D31d18']
# for subj in subj_folders:
#     subj_name = subj.split(os.sep)[-2]
#     for sub in sub_folders:
#         shutil.copytree(f'{subj}{sub}{os.sep}', os.path.join(main_folder,sub,subj_name))
#         try:
#             shutil.copy(os.path.join(subj,'MPRAGE.nii'), os.path.join(main_folder,sub,subj_name))
#         except FileNotFoundError:
#             continue
#         try:
#             shutil.copy(os.path.join(subj,'MPRAGE_brain.nii'), os.path.join(main_folder,sub,subj_name))
#         except FileNotFoundError:
#             continue
#         try:
#             shutil.copy(os.path.join(subj,'MPRAGE_brain_mask.nii'), os.path.join(main_folder,sub,subj_name))
#         except FileNotFoundError:
#             continue
#     shutil.rmtree(subj)

main_folder = r'F:\Hila\TDI\siemens'
sub_folders = ['D60d11', 'D45d13', 'D31d18']
for sub in sub_folders:
    files_2_keep = ['AxSI','cm','raw_files','streamlines',f'diff_corrected_{sub}.nii',f'diff_corrected_{sub}.bval',f'diff_corrected_{sub}.bvec',f'MPRAGE.nii',f'diff_corrected_{sub}_FA.nii.gz',f'diff_corrected_{sub}_MD.nii.gz','ADD.nii.gz']
    exp_folder = os.path.join(main_folder, sub)
    subj_folders = glob.glob(f'{exp_folder}{os.sep}[C,T]*{os.sep}')
    for subj in subj_folders:
        files = os.listdir(subj)
        for file in files:
            if file not in files_2_keep:
                os.remove(os.path.join(subj,file))

