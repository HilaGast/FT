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
    scans = glob(os.path.join(sub_dir, '*'))
    if not os.path.isdir(os.path.join(sub_dir, 'nifti')):
        os.makedirs(os.path.join(sub_dir, 'nifti'))
        for scan in scans:
            cmd = fr'"C:\Program Files\mricron\dcm2nii" -g y -o {sub_dir}\nifti {scan}'
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


def clean():
    subs = glob(r'C:\Users\admin\Desktop\subj_for_v5\*\*')
    for fold in subs:
        try:
            if 'nifti' in fold:
                os.makedirs(os.path.join(fold, 'Diffusion'))
                [os.rename(f, os.path.join(fold, 'Diffusion', os.path.split(f)[-1])) for f in glob(os.path.join(fold, '*'))
                if 'MPRAGE' not in f and os.path.isfile(f)]
                os.rename(fold, fold.replace('nifti', 'T1w'))
            elif ('T1w' not in fold) or 'MPRAGE' in fold:
                shutil.rmtree(fold)
        except FileExistsError:
            print(fold)


if __name__ == '__main__':
    copy_files()
    for sub in glob(r'C:\Users\admin\Desktop\subj_for_v5\*'):
        dicom2nii(sub)
        rename(os.path.join(sub, 'nifti'))
    clean()
