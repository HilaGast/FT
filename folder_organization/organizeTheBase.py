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
    #subj_list = ['2180','2215','2241','2252','2245','2299','2304','2629','2530','2531','2438','2477','2543','2545','2781','2703','2795','2417','2423','2478','2449','2504','2549','2493']
    #copy_files(subj_list)
    for sub in glob(r'C:\Users\Admin\Desktop\v7_calibration\thebase4ever\*'):
        dicom2nii(sub)
        clean(sub)
