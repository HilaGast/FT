
from glob import glob
import os
from fsl.file_prep import os_path_2_fsl

def rename(scan):

    files = glob(os.path.join(scan, '*'))
    print(files)
    for fi in files:
        file_parts = os.path.split(fi)

        if 'bval' in file_parts[-1]:
            os.rename(fi, os.path.join(file_parts[0], 'data.bval'))
        elif 'bvec' in file_parts[-1]:
            os.rename(fi, os.path.join(file_parts[0], 'data.bvec'))
        elif '.nii' in file_parts[-1] and 'AxCaliber3D' in file_parts[-1]:
            os.rename(fi, os.path.join(file_parts[0], 'data.nii'))

    scan_parts = os.path.split(scan)
    if 'd11.3' in scan:
        os.rename(scan, os.path.join(scan_parts[0],'d11.3D60g7.64'))
    elif 'd13.2' in scan:
        os.rename(scan, os.path.join(scan_parts[0],'d13.2D45g7.69'))
    elif 'd18' in scan:
        os.rename(scan, os.path.join(scan_parts[0],'d18D31g7.19'))


def rename_folder_files(main_path):
    for sub in glob(f'{main_path}\*'):
        for scan in glob(sub+'\*'):
            cmd = fr'"C:\Program Files\mricron\dcm2nii" -g n -o {scan} {scan}'
            print(cmd)
            os.system(cmd)

            dicoms = glob(os.path.join(scan, '*'))
            for f in dicoms:
                if f.endswith('.dcm'):
                    os.remove(f)

            rename(scan)


def bet_4_file(file_name):


    out_brain = file_name+'_brain'
    out_mask = out_brain+'_mask'
    cmd = fr'bash -lc "bet {file_name} {out_brain} -f 0.40 -g 0 -n -m'
    os.system(cmd)

    first_vol_mask = file_name[:-4]+'hifi_nodif_brain_mask'
    # save first direction file:
    cmd = fr'bash -lc "fslroi {out_mask}.nii {first_vol_mask}.nii 0 1"'
    os.system(cmd)





if __name__ == '__main__':
    main_path = r'C:\Users\Admin\Desktop\v7_calibration\Old_AxCaliber'
    #rename_folder_files(main_path)
    for sub in glob(f'{main_path}\*'):
        for scan in glob(sub+'\*'):
            for file in glob(os.path.join(scan, '*')):
                if file.endswith('.nii'):
                    file_name = file[:-4]
                    file_name = os_path_2_fsl(file_name)
                    file_name = file_name.replace(os.sep,'/')
                    bet_4_file(file_name)






