import os
current_computer = os.environ['COMP']

if current_computer == 'HOME':
    subj_folder = r'C:\Users\hila\data\subj'
    index_to_text_file = r'C:\Users\hila\data\megaatlas\megaatlas2nii.txt'
    start_with = 0

elif current_computer == 'WORK':
    #subj_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep\questionnaire'
    subj_folder = r'C:\Users\Admin\Desktop\v7_calibration\TheBase4ever'

    start_with = 0
    index_to_text_file = r'C:\Users\Admin\my_scripts\aal\yeo7_200\index2label.txt'
    #index_to_text_file = r'C:\Users\Admin\my_scripts\aal\aal3\aal2nii.txt'


elif current_computer == 'SERVER':
    subj_folder = r'F:\Hila\Language'
    #subj_folder = r'F:\Hila\Ax3D_Pack\V6\v7calibration'
    #index_to_text_file = r'F:\Hila\aal\megaatlas\megaatlas2nii.txt'
    #index_to_text_file = r'F:\Hila\aal\aal3\aal2nii.txt'
    index_to_text_file = r'F:\Hila\aal\yeo7_200\index2label.txt'
    start_with = 0


all_folders = os.listdir(subj_folder)
all_subj_folders = list()
all_subj_names = list()
for subj in all_folders[start_with::]:
    name = subj.split('_')
    name = '/'+name[1]
    name = name.replace('/',os.sep)
    all_subj_names.append(name)
    subj = '/' + subj
    subj = subj.replace('/', os.sep)
    all_subj_folders.append(subj)



