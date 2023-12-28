import os, glob
import shutil
import re
from folder_organization.organize_Old_AxCaliber_folders import bet_4_file
from fsl.file_prep import os_path_2_fsl


def copy_files(folder):
    s = glob.glob(f'{folder}{os.sep}*{os.sep}')
    if len(s) == 1:
        s = s[0]
    elif len(s) == 0:
        print('No folders in folder')
        return
    else:
        s = folder+os.sep

    folders_to_keep = ['.*AxCaliber3D.*d11.*D60.*', '.*AxCaliber3D.*d13.*D45.*','.*AxCaliber3D.*d18.*D31.*', '.*_MPRAGE.*']
    excluding_words = ['TRACEW', 'TE130','TE115']
    scans = glob.glob(f'{s}*')

    for scan in scans:
        scan_name = scan.split(os.sep)[-1]
        for f in folders_to_keep:
            reg = re.compile(f)
            sign = reg.search(scan_name)
            if sign and not any([word in scan_name for word in excluding_words]):
                shutil.copytree(scan, f'{folder}{os.sep}to_keep{os.sep}{scan_name}')

    all_folders = glob.glob(f'{folder}{os.sep}*{os.sep}')

    for f in all_folders:
        if 'to_keep' not in f:
            print(f)
            shutil.rmtree(f)

    for reamins in glob.glob(f'{folder}{os.sep}to_keep{os.sep}*'):
        scan_name = reamins.split(os.sep)[-1]
        shutil.copytree(reamins, f'{folder}{os.sep}{scan_name}')
    shutil.rmtree(f'{folder}{os.sep}to_keep')

def dicom2nii(subj):
    scans = glob.glob(f'{subj}{os.sep}*{os.sep}')
    for scan in scans:
        cmd = fr'"C:\Program Files\mricron\dcm2nii" -g n -o {subj} {scan}'
        print(cmd)
        os.system(cmd)

def rename(subj):
    scans = glob.glob(subj+os.sep+'*')
    new_folds = ['D60d11', 'D45d13', 'D31d18']
    for fol in new_folds:
        if f'{subj}{os.sep}{fol}' not in scans:
            os.mkdir(os.path.join(subj, fol))
    for scan in scans:
        scan_parts = os.path.split(scan)
        if scan_parts[-1].startswith('o') and 'MPRAGE' in scan_parts[-1]:
            os.remove(scan)
        elif scan_parts[-1].startswith('co') and 'MPRAGE' in scan_parts[-1]:
            os.remove(scan)
        elif 'D60' in scan_parts[-1] and not scan_parts[-1].endswith('mask.nii'):
            if 'bval' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'D60d11', 'diff_corrected_D60d11.bval'))
            elif 'bvec' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'D60d11', 'diff_corrected_D60d11.bvec'))
            elif '.nii' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'D60d11', 'diff_corrected_D60d11.nii'))
        elif 'D45' in scan_parts[-1] and not scan_parts[-1].endswith('mask.nii'):
            if 'bval' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'D45d13', 'diff_corrected_D45d13.bval'))
            elif 'bvec' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'D45d13', 'diff_corrected_D45d13.bvec'))
            elif '.nii' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'D45d13', 'diff_corrected_D45d13.nii'))
        elif 'D31' in scan_parts[-1] and not scan_parts[-1].endswith('mask.nii'):
            if 'bval' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'D31d18', 'diff_corrected_D31d18.bval'))
            elif 'bvec' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'D31d18', 'diff_corrected_D31d18.bvec'))
            elif '.nii' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'D31d18', 'diff_corrected_D31d18.nii'))
        elif 'MPRAGE' in scan_parts[-1]:
            os.rename(scan, os.path.join(scan_parts[0], 'MPRAGE.nii'))


def clean(subj):
    scans = glob.glob(f'{subj}{os.sep}*{os.sep}')
    for fold in scans:
        shutil.rmtree(fold)

def rename_fol(folder):
    ''' rename folder to match the naming convention: '''
    folder_name = folder.split(os.sep)[-1]
    if len(folder_name) > 8:
        subj_num = folder_name.split(' ')[0]
        if len(subj_num) == 4:
            new_subj_num = f'{subj_num[:2]}156_{subj_num[2:]}'
            new_folder = folder.replace(folder_name, new_subj_num)
            os.rename(folder, new_folder)
            return new_folder
        else:
            print('error in folder name: ', folder_name)
            return folder
    else:
        return folder



if __name__ == '__main__':
    main_folder = r'F:\Hila\TDI\siemens\more'
    all_subj = glob.glob(f'{main_folder}{os.sep}*')
    for subj in all_subj:
        # rename all folders to match the naming convention:
        subj = rename_fol(subj)
        # copy files to the main folder:
        copy_files(subj)
        # convert dicom to nii:
        dicom2nii(subj)
        # clean the folder from dicom files:
        clean(subj)
        # rename files to match the naming convention:
        rename(subj)


        for file in glob.glob(f'{subj}{os.sep}MPRAGE.nii'):
            file_name = file[:-4]
            file_name = os_path_2_fsl(file_name)
            file_name = file_name.replace(os.sep, '/')
            bet_4_file(file_name)
