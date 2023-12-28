
from glob import glob
import os
from fsl.file_prep import os_path_2_fsl
from fsl.flirt_4_oldaxcaliber_reg import *


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
    if 'd11' in scan:
        os.rename(scan, os.path.join(scan_parts[0],'d11.3D60g7.64'))
    elif 'd13' in scan:
        os.rename(scan, os.path.join(scan_parts[0],'d13.2D45g7.69'))
    elif 'd18' in scan:
        os.rename(scan, os.path.join(scan_parts[0],'d18D31g7.19'))


def rename_folder_files(main_path):
    import shutil
    for sub in glob(f'{main_path}\*'):
        for scan in glob(sub+'\*'):
            scan_name_parts = scan.split(os.sep)


            if 'CHARMED' not in scan and 'TRACEW' not in scan and '_TE95' in scan and 'AxCaliber' in scan_name_parts[-1]:
                if 'd11' in scan or 'd13' in scan or 'd18' in scan:
                    if scan_name_parts[-1].startswith('d'):
                        continue
                    else:
                        cmd = fr'"C:\Program Files\mricron\dcm2nii" -g n -o {scan} {scan}'
                        print(cmd)
                        os.system(cmd)

                        dicoms = glob(os.path.join(scan, '*'))
                        for f in dicoms:
                            if f.endswith('.dcm'):
                                os.remove(f)

                        rename(scan)

                else:
                    shutil.rmtree(scan)
            elif 'd11.3D60g7.64' in scan or 'd13.2D45g7.69' in scan or 'd18D31g7.19' in scan:
                continue
            else:
                shutil.rmtree(scan)



def bet_4_file(file_name):


    out_brain = file_name+'_brain'
    out_mask = out_brain+'_mask'
    cmd = fr'bash -lc "bet {file_name} {out_brain} -f 0.5 -g -0.1 -m"'
    os.system(cmd)

    first_vol_mask = os.path.split(file_name)[0]+'hifi_nodif_brain_mask'
    # save first direction file:
    cmd = fr'bash -lc "fslroi {out_mask}.nii {first_vol_mask}.nii 0 1"'
    os.system(cmd)





if __name__ == '__main__':

    main_path = r'F:\Hila\TDI\siemens\more'
    rename_folder_files(main_path)
    for sub in glob(f'{main_path}\*'):
        for subj_fol in glob(sub+'\*'):
            for file in glob(os.path.join(subj_fol, '*')):
                if file.endswith('.nii'):
                    file_name = file[:-4]
                    file_name = os_path_2_fsl(file_name)
                    file_name = file_name.replace(os.sep,'/')
                    bet_4_file(file_name)


            subj_folder = os_path_2_fsl(subj_fol)

            diff_file = subj_folder + '/data.nii'
            split_from_4D_2_3Ds(diff_file)
            diff_file_1st = subj_folder + '/data0000.nii'

            for diff_file_vol in glob(f'{subj_fol}\data00*.nii'):
                diff_file_vol = os_path_2_fsl(diff_file_vol)
                flirt_regis(diff_file_1st, diff_file_vol)

            merge_3Ds_2_4D(subj_folder,diff_file)
            delete_files(subj_fol)


