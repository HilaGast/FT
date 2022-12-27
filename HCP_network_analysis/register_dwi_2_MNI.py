from fsl.hcp_atlas_registration import reg_from_diff_2_mprage, hcp_subj_files, basic_files_hcp, delete_files
from fsl.file_prep import bet_4_regis_mprage, os_path_2_fsl, create_inv_mat, reg_from_mprage_2_chm_inv
import os


def flirt_primary_guess(subj_folder,moving, ref):
    options = r'-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear'

    moving_registered_flirt = subj_folder+ 'r' + moving.split(sep='/')[-1]
    moving_registered_flirt_mat = moving_registered_flirt[:-4] + '.mat'

    cmd = f'bash -lc "flirt -ref {ref} -in {moving} -out {moving_registered_flirt} -omat {moving_registered_flirt_mat} {options}"'
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return moving_registered_flirt, moving_registered_flirt_mat


def fnirt_from_subj_2_mni(subj_folder,ref, moving, moving_registered_flirt_mat):
    warp_name = subj_folder + 'atlas2subj.nii'
    cmd = f'bash -lc "fnirt --ref={ref} --in={moving} --aff={moving_registered_flirt_mat} --cout={warp_name}"'
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return warp_name


def mprage_2_dwi(subj_folder):

    subj_folder = subj_folder.replace(os.sep, '/')
    mprage_file_name, diff_file_name = hcp_subj_files(subj_folder)
    subj_folder = os_path_2_fsl(subj_folder)
    subj_mprage, out_brain = bet_4_regis_mprage(subj_folder, mprage_file_name, diff_file_name)

    ''' Registration from MPRAGE to 1st CHARMED scan using inverse matrix of CHARMED to MPRAGE registration:
    From CHARMED to MPRAGE:'''
    subj_first_charmed, out_registered, out_registered_mat = reg_from_diff_2_mprage(subj_folder, subj_mprage, diff_file_name)

    '''Creation of inverse matrix:  '''
    inv_mat = create_inv_mat(out_registered_mat)

    '''From MPRAGE to CHARMED using the inverse matrix: '''
    reg_mprage = reg_from_mprage_2_chm_inv(subj_folder, mprage_file_name, out_brain, subj_first_charmed, inv_mat)

    return subj_folder, reg_mprage

def registered_mprage_2_mni(subj_folder, mni_ref, reg_mprage):
    '''Registration from regisered MPRAGE to MNI:
            flirt for registered MPRAGE to MNI for primary guess:  '''
    moving_registered_flirt, moving_registered_flirt_mat = flirt_primary_guess(subj_folder,reg_mprage, mni_ref)

    '''fnirt for atlas based on flirt results:    '''
    warp_name = fnirt_from_subj_2_mni(subj_folder, mni_ref, reg_mprage, moving_registered_flirt_mat)

    return warp_name


def apply_mprage2mni_on_add_map(subj_folder, add_map, mni_ref, warp_name):
    '''apply fnirt warp on add map:   '''

    reg_add_map = os.path.join(subj_folder + 'r' + add_map.split(sep="\\")[-1])
    cmd = f'bash -lc "applywarp --ref={mni_ref} --in={add_map} --out={reg_add_map} --warp={warp_name} --interp=nn"'
    cmd = cmd.replace(os.sep, '/')
    os.system(cmd)


if __name__ == '__main__':
    subj, folder_name, mni_ref = basic_files_hcp(False)[:3]
    for s in subj[338::]:
        if os.path.exists(s+'raverage_add_map.nii'):
             continue
        else:
             add_map = os_path_2_fsl(os.path.join(s,'average_add_map.nii'))

        subj_folder, reg_mprage = mprage_2_dwi(s)
        warp_name = registered_mprage_2_mni(subj_folder, mni_ref, reg_mprage)

        apply_mprage2mni_on_add_map(subj_folder, add_map, mni_ref, warp_name)

            # except UnboundLocalError:
            #     continue
        delete_files(s)