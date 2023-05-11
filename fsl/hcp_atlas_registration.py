from fsl.file_prep import bet_4_regis_mprage, os_path_2_fsl, create_inv_mat, reg_from_mprage_2_chm_inv, flirt_primary_guess, fnirt_from_atlas_2_subj, apply_fnirt_warp_on_label
import os, glob


def reg_from_diff_2_mprage(subj_folder,subj_mprage, diff_file_name):

    cmd = fr'bash -lc "fslroi {subj_folder}/{diff_file_name} {subj_folder}/data_1st 0 1"'
    os.system(cmd)
    subj_first_charmed = subj_folder + '/data_1st.nii.gz'
    out_registered = subj_folder + '/rdata_1st.nii.gz'
    out_registered_mat = out_registered[:-7] +'.mat'
    options = '-bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'
    cmd = 'bash -lc "flirt -ref {0} -in {1} -out {2} -omat {3} {4}"'.format(subj_mprage, subj_first_charmed, out_registered, out_registered_mat, options)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return subj_first_charmed, out_registered, out_registered_mat


def hcp_subj_files(subj_folder):
    for file in os.listdir(subj_folder):
        if 'MPRAGE' in file and not (file.startswith('r') or 'brain' in file):
            mprage_file_name = file
        if 'data.nii' in file:
            diff_file_name = file
    return mprage_file_name, diff_file_name


def basic_files_hcp(atlas = 'bna', cortex_only=False):
    if atlas == 'bna':
        if cortex_only:
            atlas_label = r'G:\data\atlases\BNA\newBNA_Labels.nii'
            atlas_template = r'G:\data\atlases\BNA\MNI152_T1_1mm.nii'

        else:
            atlas_label = r'G:\data\atlases\BNA\BN_Atlas_274_combined_1mm.nii'
            atlas_template = r'G:\data\atlases\BNA\MNI152_T1_1mm.nii'

    elif atlas == 'megaatlas':
        atlas_label = r'G:\data\atlases\megaatlas\MegaAtlas_cortex_Labels.nii'
        atlas_template = r'G:\data\atlases\megaatlas\Schaefer_template.nii'

    elif atlas == 'yeo7_200':
        atlas_label = r'G:\data\atlases\yeo\yeo7_200\yeo7_200_atlas.nii'
        atlas_template = r'G:\data\atlases\yeo\yeo7_200\Schaefer_template.nii'

    elif atlas == 'yeo7_100':
        atlas_label = r'G:\data\atlases\yeo\yeo7_100\yeo7_100_atlas.nii'
        atlas_template = r'G:\data\atlases\yeo\yeo7_100\Schaefer_template.nii'


    atlas_label = os_path_2_fsl(atlas_label)
    atlas_template = os_path_2_fsl(atlas_template)

    folder_name = r'G:\data\V7\HCP'
    subj = glob.glob(f'{folder_name}{os.sep}*[0-9]{os.sep}')

    return subj, folder_name, atlas_template, atlas_label


def register_atlas_2_diff(subj_folder, atlas_template, atlas_label):

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
    out_registered = reg_from_mprage_2_chm_inv(subj_folder, mprage_file_name, out_brain, subj_first_charmed, inv_mat)

    '''Registration from atlas to regisered MPRAGE:
        flirt for atlas to registered MPRAGE for primary guess:  '''
    atlas_brain, atlas_registered_flirt, atlas_registered_flirt_mat = flirt_primary_guess(subj_folder, atlas_template,
                                                                                          out_registered)

    '''fnirt for atlas based on flirt results:    '''
    warp_name = fnirt_from_atlas_2_subj(subj_folder, out_registered, atlas_brain, atlas_registered_flirt_mat,
                                        cortex_only=False)

    '''apply fnirt warp on atlas labels:   '''
    atlas_labels_registered = apply_fnirt_warp_on_label(subj_folder, atlas_label, out_registered, warp_name)


def delete_files(s):
    files = []
    #files.append(s+'rMPRAGE_brain.nii')
    files.append(s+'rdata_1st_inv.mat')
    files.append(s+'rdata_1st.mat')
    files.append(s+'rdata_1st.nii')
    files.append(s+'rMNI152_T1_1mm_brain.mat')
    files.append(s+'rSchaefer_template.mat')

    for fi in files:
        if os.path.exists(fi):
            os.remove(fi)
    if os.path.exists(s+'data_1st.nii') and os.path.exists(s+'data_1st.nii.gz'):
        os.remove(s+'data_1st.nii')


if __name__ == '__main__':
    subj, folder_name, atlas_template, atlas_label = basic_files_hcp(atlas = 'yeo7_100', cortex_only=True)
    for s in subj[::]:
        # if os.path.exists(s+'ryeo7_200_atlas.nii'):
        #if os.path.exists(s+'rMegaAtlas_cortex_Labels.nii'):
        #if os.path.exists(s + 'rBN_Atlas_274_combined_1mm.nii'):
        if os.path.exists(s + 'ryeo7_100_atlas.nii'):
            continue
        else:
            try:
                register_atlas_2_diff(s, atlas_template, atlas_label)

            except UnboundLocalError:
                continue
        delete_files(s)