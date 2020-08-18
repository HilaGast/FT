import os
current_computer = os.environ['COMP']
if current_computer == 'HOME':
    subj_folder = r'C:\Users\hila\data\subj'
    start_with = 0
elif current_computer == 'WORK':
    subj_folder = r'F:\Hila\Ax3D_Pack\V6\after_file_prep'
    start_with = 0
elif current_computer == 'SERVER':
    subj_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep\questionnaire'
    start_with = 0

all_folders = os.listdir(subj_folder)
all_subj_folders = list()
all_subj_names = list()
for subj in all_folders[start_with::]:
    name = subj.split('_')
    name = '/'+name[3]
    name = name.replace('/',os.sep)
    all_subj_names.append(name)
    subj = '/' + subj
    subj = subj.replace('/', os.sep)
    all_subj_folders.append(subj)



