import os, glob
import shutil
import re
from folder_organization.organize_Old_AxCaliber_folders import bet_4_file
from fsl.file_prep import os_path_2_fsl


def copy_files(subj):
    s = glob.glob(f'{subj}{os.sep}*{os.sep}')[0]
    folders = ['.*d11.3D60$', '.*d13.2D45$','.*d18D31$', '.*_MPRAGE.*']
    scans = glob.glob(f'{s}*')
    for scan in scans:
        scan_name = scan.split(os.sep)[-1]
        for f in folders:
            reg = re.compile(f)
            sign = reg.search(scan_name)
            if sign:
                shutil.copytree(scan, f'{subj}{os.sep}{scan_name}')

    shutil.rmtree(s)

def dicom2nii(subj):
    scans = glob.glob(f'{subj}{os.sep}*{os.sep}')
    for scan in scans:
        cmd = fr'"C:\Program Files\mricron\dcm2nii" -g n -o {subj} {scan}'
        print(cmd)
        os.system(cmd)

def rename(subj):
    scans = glob.glob(subj+os.sep+'*')
    #print(scans)
    for scan in scans:
        scan_parts = os.path.split(scan)
        if 'D60' in scan_parts[-1] and not scan_parts[-1].endswith('mask.nii'):
            if 'bval' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'diff_corrected_D60d11.bval'))
            elif 'bvec' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'diff_corrected_D60d11.bvec'))
            elif '.nii' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'diff_corrected_D60d11.nii'))
        elif 'D45' in scan_parts[-1] and not scan_parts[-1].endswith('mask.nii'):
            if 'bval' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'diff_corrected_D45d13.bval'))
            elif 'bvec' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'diff_corrected_D45d13.bvec'))
            elif '.nii' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'diff_corrected_D45d13.nii'))
        elif 'D31' in scan_parts[-1] and not scan_parts[-1].endswith('mask.nii'):
            if 'bval' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'diff_corrected_D31d18.bval'))
            elif 'bvec' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'diff_corrected_D31d18.bvec'))
            elif '.nii' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'diff_corrected_D31d18.nii'))
        elif 'MPRAGE' in scan_parts[-1]:
            os.rename(scan, os.path.join(scan_parts[0], 'MPRAGE.nii'))


def clean(subj):
    scans = glob.glob(f'{subj}{os.sep}*{os.sep}')
    for fold in scans:
        shutil.rmtree(fold)


if __name__ == '__main__':
    main_folder = r'F:\Hila\TDI\siemens\more'
    all_subj = glob.glob(f'{main_folder}{os.sep}*')
    #all_subj = [all_subj[38],all_subj[45],all_subj[48],all_subj[52],all_subj[54],all_subj[55]]
    for subj in all_subj[2:]:
        #copy_files(subj)
        dicom2nii(subj)
        clean(subj)
        rename(subj)
        # for file in glob.glob(f'{subj}{os.sep}diff_corrected_D31d18.nii'):
        #     file_name = file[:-4]
        #     file_name = os_path_2_fsl(file_name)
        #     file_name = file_name.replace(os.sep, '/')
        #     bet_4_file(file_name)
