import os
from fsl.file_prep import os_path_2_fsl, basic_files

def vol_file(subj_folder,vol_type):

    for file in os.listdir(subj_folder):
        if file.endswith('.nii') and vol_type in file:
            vol_file_name = file

    return vol_file_name


def template_file(subj_folder):
    for file in os.listdir(subj_folder):
        if file.endswith('_T2.nii'):
            template_file_name = file

    return template_file_name


def apply_fnirt_on_vol(s, folder_name, vol_type, template_brain, warp_name):
    subj_name = r'/' + s + r'/'
    subj_folder = folder_name + subj_name
    subj_folder = subj_folder.replace(os.sep, '/')

    vol = vol_file(subj_folder,vol_type)

    subj_folder = os_path_2_fsl(subj_folder)
    vol = os.path.join(subj_folder,vol)

    # apply fnirt:
    vol_registered = os.path.join(subj_folder+ 'rr' + vol.split(sep="/")[-1]) ##
    cmd = 'bash -lc "applywarp --ref={0} --in={1} --out={2} --warp={3} --interp={4}"'.format(template_brain, vol, vol_registered, warp_name, 'nn')
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)


def apply_flirt_on_vol(s, folder_name, vol_type, template_brain, warp_name):
    subj_name = r'/' + s + r'/'
    subj_folder = folder_name + subj_name
    subj_folder = subj_folder.replace(os.sep, '/')

    vol = vol_file(subj_folder,vol_type)

    subj_folder = os_path_2_fsl(subj_folder)
    vol = os.path.join(subj_folder,vol)

    # apply fnirt:
    vol_registered = os.path.join(subj_folder+ 'rr' + vol.split(sep="/")[-1]) ##
    cmd = 'bash -lc "flirt -in {0} -ref {1} -out {2} -applyxfm -init {3}"'.format(vol, template_brain, vol_registered, warp_name)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)


def reg_vols_2_atlas(s, folder_name):
    subj_name = r'/' + s + r'/'
    subj_folder = folder_name + subj_name
    subj_folder = subj_folder.replace(os.sep, '/')

    # search file name:
    template_brain = template_file(subj_folder)
    diff_file = 'diff_corrected_b2000.nii'

    subj_folder = os_path_2_fsl(subj_folder)

    template_brain = os.path.join(subj_folder,template_brain)
    diff_file = os.path.join(subj_folder,diff_file)

    # flirt for primary guess:
    options = r'-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear'

    diff_flirt = os.path.join(subj_folder + diff_file.split(sep="/")[-1]+'_reg')
    diff_flirt_mat = diff_flirt[:-4] + '.mat'

    cmd = 'bash -lc "flirt -ref {0} -in {1} -out {2} -omat {3} {4}"'.format(template_brain, diff_file, diff_flirt, diff_flirt_mat, options)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    # apply flirt on 4D:
    cmd = 'bash -lc "applyxfm4D {0} {1} {2} {3} -singlematrix"'.format(diff_file, template_brain, diff_flirt, diff_flirt_mat)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    #warp_name = diff_file[:-4] + '_diff2template.nii'

    #cmd = 'bash -lc "fnirt --ref={0} --in={1} --aff={2} --cout={3}"'.format(template_brain, diff_file, diff_flirt_mat, warp_name)
    #cmd = cmd.replace(os.sep,'/')
    #os.system(cmd)

    return template_brain, diff_flirt_mat



if __name__ == '__main__':
    from multiprocessing import Process
    subj, folder_name = basic_files(False, atlas_type='yeo7_200')[0:2]
    for s in [subj[8],subj[11]]:
        template_brain, warp_name = reg_vols_2_atlas(s, folder_name)
        vol_type = 'diff_corrected_3_2_AxFr7'
        apply_flirt_on_vol(s,folder_name,vol_type,template_brain,warp_name)
        vol_type = 'diff_corrected_3_2_AxPasi7'
        apply_flirt_on_vol(s,folder_name,vol_type,template_brain,warp_name)
        vol_type = '_ADD_along_streamlines'
        apply_flirt_on_vol(s,folder_name,vol_type,template_brain,warp_name)

