import os
subj_folder = r'C:\Users\hila\data\subj'
#subj_folder = r'D:\after_file_prep'
all_folders = os.listdir(subj_folder)
all_subj_folders = list()
all_subj_names = list()
for subj in all_folders:
    name = subj.split('_')
    name = '/'+name[3]
    name = name.replace('/',os.sep)
    all_subj_names.append(name)
    subj = '/' + subj
    subj = subj.replace('/', os.sep)
    all_subj_folders.append(subj)



