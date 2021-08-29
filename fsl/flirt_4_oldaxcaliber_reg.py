import os
from fsl.file_prep import os_path_2_fsl
from glob import glob



def split_from_4D_2_3Ds(diff_file):
    cmd = fr'bash -lc "fslsplit {diff_file} {diff_file} -t"'
    cmd = cmd.replace(os.sep, '/')
    os.system(cmd)


def flirt_regis(diff_file_1st, diff_file_vol):
    out_registered = diff_file_vol[:-4] +'_r.nii'
    options = '-bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'

    cmd = rf'bash -lc "flirt -ref {diff_file_1st} -in {diff_file_vol} -out {out_registered} {options}"'
    cmd = cmd.replace(os.sep, '/')
    os.system(cmd)


def merge_3Ds_2_4D(subj_folder,diff_file):
    registered_diff = subj_folder + '/diff_corrected.nii'
    sep_files = diff_file[:-4]+'00*_r.nii'
    cmd = fr'bash -lc "fslmerge -t {registered_diff} {sep_files}"'
    cmd = cmd.replace(os.sep, '/')
    os.system(cmd)


def delete_files(subj_fol):
    scans = glob(subj_fol+ r'\data00*.nii')
    for fold in scans:
        os.remove(fold)

if __name__ == '__main__':
    main_path = r'C:\Users\Admin\Desktop\v7_calibration\Old_AxCaliber'
    for sub in glob(f'{main_path}\*'):
        for subj_fol in glob(sub+'\*'):
            subj_folder = os_path_2_fsl(subj_fol)

            diff_file = subj_folder + '/data.nii'
            split_from_4D_2_3Ds(diff_file)
            diff_file_1st = subj_folder + '/data0000.nii'


            for diff_file_vol in glob(f'{subj_fol}\data00*.nii'):
                diff_file_vol = os_path_2_fsl(diff_file_vol)
                flirt_regis(diff_file_1st, diff_file_vol)

            merge_3Ds_2_4D(subj_folder,diff_file)
            delete_files(subj_fol)