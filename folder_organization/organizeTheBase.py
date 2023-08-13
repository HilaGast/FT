from glob import glob
import os
import shutil
import re

def copy_files(subj_list):
    subs = []
    for s in subj_list:
        try:
            subs.append(glob(f'{os.sep}{os.sep}132.66.46.223{os.sep}YA_Shared{os.sep}*{os.sep}YA_lab_Yaniv_00{s}*')[0])
        except IndexError:
            try:
                subs.append(glob(f'{os.sep}{os.sep}132.66.46.223{os.sep}YA_Shared{os.sep}*{os.sep}*{os.sep}YA_lab_Yaniv_00{s}*')[0])
            except IndexError:
                break

    folders = ['.*d15D45_AP$', '.*d15D45_PA$', '.*MPRAGE_RL$']
    for sub in subs:
        print(sub)
        for scan in glob(f'{sub}{os.sep}*'):
            scan_name = scan.split(os.sep)[-1]
            for f in folders:
                reg = re.compile(f)
                sign = reg.search(scan_name)
                if sign:
                    shutil.copytree(scan, os.path.join(r'C:\Users\Admin\Desktop\v7_calibration\thebase4ever',scan.split(os.sep)[-2], scan_name))

def copy_more_tracts_file(subj_list, dest_dir):
    for sub in subj_list:
        sub_name = sub.split(os.sep)[-1]
        dest_name = os.path.join(dest_dir, sub_name,'streamlines')
        if not os.path.exists(dest_name):
            os.mkdir(dest_name)
        for tract_file in glob(os.path.join(sub, '*.tck')):
            tract_name = tract_file.split(os.sep)[-1]
            if not os.path.exists(os.path.join(dest_name, tract_name)):
                shutil.copy(tract_file, dest_name)

def dicom2nii(sub_dir):
    scans = glob(os.path.join(sub_dir, '*/'))
    #folders = ['d15D45_AP', 'd15D45_PA', 'MPRAGE']
    for scan in scans:
    #    if folders[0] in scan or folders[1] in scan or folders[2] in scan or folders[3] in scan:
        cmd = fr'"C:\Program Files\mricron\dcm2nii" -g n -o {sub_dir} {scan}'
        print(cmd)
        os.system(cmd)

def rename(nifti_dir):
    scans = glob(os.path.join(nifti_dir, '*'))
    print(scans)
    for scan in scans:
        scan_parts = os.path.split(scan)
        if 'd15D45AP' in scan_parts[-1]:
            if 'bval' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'data.bval'))
            elif 'bvec' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'data.bvec'))
            elif '.nii' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'data.nii'))
        elif 'd15D45PA' in scan_parts[-1]:
            if '.nii' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'data_PA.nii'))
            else:
                os.remove(scan)
        elif 'MPRAGE' in scan_parts[-1]:
            os.rename(scan, os.path.join(scan_parts[0], 'MPRAGE.nii'))


def clean(sub_dir):
    scans = glob(os.path.join(sub_dir, '*/'))
    for fold in scans:
        shutil.rmtree(fold)

def out_from_axsi_folder(sub):
    if os.path.exists(f'{sub}{os.sep}AxSI'):
        for file in glob(f'{sub}{os.sep}AxSI{os.sep}*'):
            shutil.move(file, sub)
        shutil.rmtree(f'{sub}{os.sep}AxSI')


def move_streamlines_into_fol(sub):
    dir_name = f'{sub}{os.sep}streamlines'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for file in os.listdir(sub):
        if '.tck' in file or '.trk' in file and not os.path.exists(os.path.join(dir_name, file)):
            shutil.move(os.path.join(sub,file), dir_name)
        elif '.tck' in file or '.trk' in file and os.path.exists(os.path.join(dir_name, file)):
            os.remove(os.path.join(sub,file))



if __name__ == '__main__':

    subj_list = glob(r'Y:\qnap_hcp\TheBase4Ever4Hila\*[0-9]')
    dest_fold = r'F:\Hila\TDI\TheBase4Ever'
    copy_more_tracts_file(subj_list, dest_fold)
    # for sub in glob(r'F:\Hila\TDI\TheBase4Ever\*')[::]:
    #     #dicom2nii(sub)
    #     #clean(sub)
    #     #rename(sub)
    #     out_from_axsi_folder(sub)
    #     move_streamlines_into_fol(sub)
