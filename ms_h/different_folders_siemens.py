import os, glob, shutil

main_fol = 'Y:\qnap\siemens'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')

for fol in all_subj_fol:
    try:
        os.mkdir(f'{fol}D31d18')
    except FileExistsError:
        print('Folder already exists')
    try:
        os.mkdir(f'{fol}D45d13')
    except FileExistsError:
        print('Folder already exists')
    try:
        os.mkdir(f'{fol}D60d11')
    except FileExistsError:
        print('Folder already exists')

    files = (file for file in os.listdir(fol) if os.path.isfile(os.path.join(fol, file)))
    for file in files:
        if 'D31d18' in file:
            shutil.move(f'{fol}{file}',f'{fol}D31d18')
        elif 'D45d13' in file:
            shutil.move(f'{fol}{file}',f'{fol}D45d13')
        elif 'D60d11' in file:
            shutil.move(f'{fol}{file}',f'{fol}D60d11')

