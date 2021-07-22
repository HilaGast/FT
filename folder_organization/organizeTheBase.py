from glob import glob
import os
import shutil


def copy_files():
    subs = glob(r'Y:\*')
    for sub in subs:
        print(sub)
        folders = ['05_ep2d_d15.5D60_MB3_AP', '06_ep2d_d15.5D60_MB3_PA', '03_T1w_MPRAGE_RL']
        for f in folders:
            try:
                shutil.copytree(os.path.join(sub, f), os.path.join(sub.replace(r'Y:', r'C:\Users\admin\Desktop\subj_for_v5'), f))
            except FileNotFoundError:
                break


def dicom2nii(sub_dir):
    scans = glob(os.path.join(sub_dir, '*/'))
    folders = ['ep2d_d15.5D60_MB3_AP', 'ep2d_d15.5D60_MB3_PA', 'MPRAGE', 'd15D45']
    for scan in scans:
        if folders[0] in scan or folders[1] in scan or folders[2] in scan or folders[3] in scan:
            cmd = fr'"C:\Program Files\mricron\dcm2nii" -g n -o {sub_dir} {scan}'
            print(cmd)
            os.system(cmd)

def rename(nifti_dir):
    scans = glob(os.path.join(nifti_dir, '*'))
    print(scans)
    for scan in scans:
        scan_parts = os.path.split(scan)
        if 'MB3AP' in scan_parts[-1]:
            if 'bval' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'bvals'))
            elif 'bvec' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'bvecs'))
            elif '.nii' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'dif_AP.nii'))
        elif 'MB3PA' in scan_parts[-1]:
            if '.nii' in scan_parts[-1]:
                os.rename(scan, os.path.join(scan_parts[0], 'dif_PA.nii'))
        elif 'MPRAGE' in scan_parts[-1]:
            os.rename(scan, os.path.join(scan_parts[0], 'MPRAGE.nii'))


def clean(sub_dir):
    scans = glob(os.path.join(sub_dir, '*/'))
    for fold in scans:
        shutil.rmtree(fold)



if __name__ == '__main__':
    #copy_files()
    for sub in glob(r'C:\Users\Admin\Desktop\Language\*'):
        dicom2nii(sub)
        clean(sub)
        #rename(os.path.join(sub, 'nifti'))
