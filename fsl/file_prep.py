import os
from fsl.eddy_correct_diff import eddy_corr


def basic_files(cortex_only=True,atlas_type='mega'):
    if atlas_type == 'mega':
        if cortex_only:
            atlas_label = r'C:\Users\Admin\my_scripts\aal\megaatlas\MegaAtlas_cortex_Labels.nii'
        else:
            atlas_label = r'C:\Users\Admin\my_scripts\aal\megaatlas\MegaAtlas_Labels_highres.nii'

        atlas_template = r'C:\Users\Admin\my_scripts\aal\megaatlas\Schaefer_template.nii'

    elif atlas_type == 'aal3':
        atlas_label = r'F:\Hila\aal\aal3\AAL3_highres_atlas.nii'
        atlas_template = r'F:\Hila\aal\aal3\AAL3_highres_template.nii'
        #atlas_label = r'F:\Hila\aal\aal3\registered\AAL3_highres_atlas_corrected.nii'
        #atlas_template = r'F:\Hila\aal\aal3\registered\MNI152_T1_1mm.nii'


    elif atlas_type == 'yeo7_200':
        atlas_label = r'F:\Hila\aal\yeo7_200\yeo7_200_atlas.nii'
        atlas_template = r'F:\Hila\aal\yeo7_200\Schaefer_template.nii'
        #atlas_label = r'F:\Hila\aal\aal3\registered\AAL3_highres_atlas_corrected.nii'
        #atlas_template = r'F:\Hila\aal\aal3\registered\MNI152_T1_1mm.nii'


    atlas_label = os_path_2_fsl(atlas_label)
    atlas_template = os_path_2_fsl(atlas_template)

    #folder_name = r'F:\Hila\Ax3D_Pack\V6\after_file_prep'
    folder_name = r'F:\Hila\balance\ec\after'
    all_subj_folders = os.listdir(folder_name)
    subj = all_subj_folders

    return subj, folder_name, atlas_template, atlas_label


def subj_files(subj_folder):

    for file in os.listdir(subj_folder):
        if 'wMPRAGERL' in file and not (file.startswith('r') or 'brain' in file):
            mprage_file_name = file
        if file.endswith('001.nii') and 'AP' in file:
            diff_file_name = file
        if file.endswith('001.nii') and 'PA' in file:
            pa_file_name = file
    return mprage_file_name, diff_file_name, pa_file_name


def bet_4_regis_mprage(subj_folder,mprage_file_name):

    subj_mprage = subj_folder + mprage_file_name
    # BET for registered MPRAGE:
    out_brain = subj_mprage[:-4]+'_brain'

    cmd = 'bash -lc "bet {0} {1} {2} {3}"'.format(subj_mprage[:-4], out_brain,'-f 0.30','-g 0.20')
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)
    # save first corrected diff:
    cmd = fr'bash -lc "fslroi {subj_folder}/diff_corrected.nii {subj_folder}/diff_corrected_1st 0 1"'
    os.system(cmd)

    return subj_mprage, out_brain


def reg_from_chm_2_mprage(subj_folder,subj_mprage):
    subj_first_charmed = subj_folder + '/diff_corrected_1st.nii'
    out_registered = subj_folder + '/rdiff_corrected_1st.nii'
    out_registered_mat = out_registered[:-4] +'.mat'
    options = '-bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'

    cmd = 'bash -lc "flirt -ref {0} -in {1} -out {2} -omat {3} {4}"'.format(subj_mprage, subj_first_charmed, out_registered, out_registered_mat, options)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return subj_first_charmed, out_registered, out_registered_mat


def create_inv_mat(out_registered_mat):
    inv_mat = out_registered_mat[:-4] + '_inv.mat'
    cmd = 'bash -lc "convert_xfm -omat {0} -inverse {1}"'.format(inv_mat, out_registered_mat)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return inv_mat


def reg_from_mprage_2_chm_inv(subj_folder, mprage_file_name, out_brain, subj_first_charmed, inv_mat):
    out_registered = subj_folder + 'r' + mprage_file_name[:-4]+'_brain.nii'
    cmd = 'bash -lc "flirt -in {0} -ref {1} -out {2} -applyxfm -init {3}"'.format(out_brain, subj_first_charmed, out_registered, inv_mat)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return out_registered


def bet_4_atlas(atlas_template):
    atlas_brain = atlas_template[:-4] + '_brain'
    cmd = 'bash -lc "bet {0} {1} {2} {3}"'.format(atlas_template[:-4], atlas_brain,'-f 0.45','-g -0.1')
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return atlas_brain


def flirt_primary_guess(subj_folder,atlas_template, out_registered):
    options = r'-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear'
    atlas_brain = atlas_template[:-4] + '_brain.nii'

    atlas_registered_flirt = os.path.join(subj_folder+ 'r' + atlas_brain.split(sep="\\")[-1])
    atlas_registered_flirt_mat = atlas_registered_flirt[:-4] + '.mat'

    cmd = 'bash -lc "flirt -ref {0} -in {1} -out {2} -omat {3} {4}"'.format(out_registered, atlas_brain, atlas_registered_flirt, atlas_registered_flirt_mat, options)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return atlas_brain, atlas_registered_flirt, atlas_registered_flirt_mat


def fnirt_from_atlas_2_subj(subj_folder,out_registered, atlas_brain, atlas_registered_flirt_mat, cortex_only = True):
    if cortex_only:
        warp_name = subj_folder + 'atlas2subjmegaatlas.nii'
    else:
        warp_name = subj_folder + 'atlas2subj.nii'
    cmd = 'bash -lc "fnirt --ref={0} --in={1} --aff={2} --cout={3}"'.format(out_registered, atlas_brain, atlas_registered_flirt_mat, warp_name)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return warp_name


def apply_fnirt_warp_on_template(subj_folder, atlas_brain, out_registered, warp_name):
    atlas_registered = os.path.join(subj_folder+ 'rr' + atlas_brain.split(sep="\\")[-1])
    cmd = 'bash -lc "applywarp --ref={0} --in={1} --out={2} --warp={3} --interp={4}"'.format(out_registered, atlas_brain, atlas_registered, warp_name, 'nn')
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return atlas_registered


def apply_fnirt_warp_on_label(subj_folder, atlas_label, out_registered, warp_name):
    atlas_labels_registered = os.path.join(subj_folder+ 'r' + atlas_label.split(sep="\\")[-1])
    cmd = 'bash -lc "applywarp --ref={0} --in={1} --out={2} --warp={3} --interp={4}"'.format(out_registered, atlas_label, atlas_labels_registered, warp_name, 'nn')
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    return atlas_labels_registered


def apply_fnirt_warp_on_masks(subj_folder, atlas_mask, out_registered, warp_name):
    atlas_mask_registered = os.path.join(subj_folder+ 'r' + atlas_mask.split(sep="\\")[-1])
    cmd = 'bash -lc "applywarp --ref={0} --in={1} --out={2} --warp={3} --interp={4}"'.format(out_registered, atlas_mask, atlas_mask_registered, warp_name, 'nn')
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)


def fast_seg(out_registered):
    options = r'-t 1 -n 3 -H 0.1 -I 4 -l 10.0 -o'
    cmd = 'bash -lc "fast {0} {1} {2}"'.format(options, out_registered, out_registered)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)


def os_path_2_fsl(path):
    if 'F:' in path:
        path = path.replace('F:','/mnt/f')
    elif 'C:' in path:
        path = path.replace('C:', '/mnt/c')
    elif 'D:' in path:
        path = path.replace('D:', '/mnt/d')

    return path


def all_func_to_run(s, folder_name, atlas_template, atlas_label):
    subj_name = r'/' + s + r'/'
    subj_folder = folder_name + subj_name
    subj_folder = subj_folder.replace(os.sep, '/')

    mprage_file_name, diff_file_name, pa_file_name = subj_files(subj_folder)

    subj_folder = os_path_2_fsl(subj_folder)

    #eddy_corr(subj_folder,diff_file_name,pa_file_name)

    subj_mprage, out_brain = bet_4_regis_mprage(subj_folder, mprage_file_name)

    ''' Registration from MPRAGE to 1st CHARMED scan using inverse matrix of CHARMED to MPRAGE registration:
    From CHARMED to MPRAGE:'''
    subj_first_charmed, out_registered, out_registered_mat = reg_from_chm_2_mprage(subj_folder, subj_mprage)

    '''Creation of inverse matrix:  '''
    inv_mat = create_inv_mat(out_registered_mat)

    '''From MPRAGE to CHARMED using the inverse matrix: '''
    out_registered = reg_from_mprage_2_chm_inv(subj_folder, mprage_file_name, out_brain, subj_first_charmed, inv_mat)

    ''' BET for mni template:
        BET for mni template:
        if not performed before, run:   '''
    # atlas_brain = bet_4_atlas(atlas_template)

    '''Registration from megaatlas to regisered MPRAGE:
        flirt for megaatlas to registered MPRAGE for primary guess:  '''
    atlas_brain, atlas_registered_flirt, atlas_registered_flirt_mat = flirt_primary_guess(subj_folder, atlas_template,
                                                                                          out_registered)

    '''fnirt for megaatlas based on flirt results:    '''
    warp_name = fnirt_from_atlas_2_subj(subj_folder, out_registered, atlas_brain, atlas_registered_flirt_mat,
                                        cortex_only=False)

    '''apply fnirt warp on atlas template:  '''
    atlas_registered = apply_fnirt_warp_on_template(subj_folder, atlas_brain, out_registered, warp_name)

    '''apply fnirt warp on atlas labels:   '''
    atlas_labels_registered = apply_fnirt_warp_on_label(subj_folder, atlas_label, out_registered, warp_name)

    '''FAST segmentation:   '''
    #fast_seg(out_registered)

    print('Finished file prep for ' + subj_name[:-1])


if __name__ == '__main__':
    from multiprocessing import Process
    subj, folder_name, atlas_template, atlas_label = basic_files(False, atlas_type='yeo7_200')
    for s in subj[::]:
        all_func_to_run(s, folder_name, atlas_template, atlas_label)


    '''multi procesing:'''
    '''i= [41,42,43,44,46,47,50,53]
    subj= [subj[j] for j in i]
    process = [Process(target=all_func_to_run,args=(s, folder_name, atlas_template, atlas_label)) for s in subj]
    for p in process:
        p.start()
    for p in process:
        p.join()
    '''