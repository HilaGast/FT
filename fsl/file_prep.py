from FT.all_subj import all_subj_names
import os
from FT.fsl.eddy_correct_diff import eddy_corr

#for megaatlas:
atlas_template = r'C:\Users\Admin\my_scripts\aal\megaatlas\MegaAtla_Template.nii'
atlas_template = atlas_template.replace('C:', '/mnt/c')
atlas_label = r'C:\Users\Admin\my_scripts\aal\megaatlas\MegaAtlas_cortex_Labels.nii'
atlas_label = atlas_label.replace('C:', '/mnt/c')

main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5/'
folder_name = main_folder
main_folder = main_folder.replace('C:', '/mnt/c')

subj = all_subj_names[16::]
for s in subj:
    subj_name = s[1::] + r'/'
    subj_folder = folder_name + subj_name
    subj_folder = subj_folder.replace(os.sep,'/')

    for file in os.listdir(subj_folder):
        if 'wMPRAGERL' in file and not file.startswith('r'):
            mprage_file_name = file
        if file.endswith('001.nii') and 'AP' in file:
            diff_file_name = file
        if file.endswith('001.nii') and 'PA' in file:
            pa_file_name = file

    subj_folder = subj_folder.replace('C:', '/mnt/c')

    ''' FINISH IT!'''

    eddy_corr(subj_folder,diff_file_name,pa_file_name) #################################################################




    subj_mprage = subj_folder + mprage_file_name
    # BET for registered MPRAGE:
    out_brain = subj_mprage[:-4]+'_brain'

    cmd = 'bash -lc "bet {0} {1} {2} {3}"'.format(subj_mprage[:-4], out_brain,'-f 0.30','-g 0.20')
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)
    # save first corrected diff:
    cmd = fr'bash -lc "fslroi {subj_folder}/diff_corrected.nii {subj_folder}/diff_corrected_1st 0 1"'
    os.system(cmd)

    ## Registration from MPRAGE to 1st CHARMED scan using inverse matrix of CHARMED to MPRAGE registration:
    # From CHARMED to MPRAGE:
    subj_first_charmed = subj_folder + '/diff_corrected_1st.nii'
    out_registered = subj_folder + '/rdiff_corrected_1st.nii'
    out_registered_mat = out_registered[:-4] +'.mat'
    options = '-bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12'

    cmd = 'bash -lc "flirt -ref {0} -in {1} -out {2} -omat {3} {4}"'.format(subj_mprage, subj_first_charmed, out_registered, out_registered_mat, options)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    # Creation of inverse matrix:
    inv_mat = out_registered_mat[:-4] + '_inv.mat'
    cmd = 'bash -lc "convert_xfm -omat {0} -inverse {1}"'.format(inv_mat, out_registered_mat)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    # From MPRAGE to CHARMED using the inverse matrix:
    out_registered = subj_folder + 'r' + mprage_file_name[:-4]+'_brain.nii'
    cmd = 'bash -lc "flirt -in {0} -ref {1} -out {2} -applyxfm -init {3}"'.format(out_brain, subj_first_charmed, out_registered, inv_mat)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    ## BET for mni template:

    # BET for mni template:


# if not performed before, run:
#atlas_brain = atlas_template[:-4] + '_brain'
#cmd = 'bash -lc "bet {0} {1} {2} {3}"'.format(atlas_template[:-4], atlas_brain,'-f 0.45','-g -0.1')
#cmd = cmd.replace(os.sep,'/')
#os.system(cmd)

## Registration from MNI to regisered MPRAGE:

# flirt for MNI to registered MPRAGE for primary guess:
    options = r'-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear'
    atlas_brain = atlas_template[:-4] + '_brain.nii'

    atlas_registered_flirt = os.path.join(subj_folder+ 'r' + atlas_brain.split(sep="\\")[-1])
    atlas_registered_flirt_mat = atlas_registered_flirt[:-4] + '.mat'

    cmd = 'bash -lc "flirt -ref {0} -in {1} -out {2} -omat {3} {4}"'.format(out_registered, atlas_brain, atlas_registered_flirt, atlas_registered_flirt_mat, options)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    # fnirt for MNI based on flirt results:
    #warp_name = subj_folder + 'atlas2subj.nii'
    warp_name = subj_folder + 'atlas2subjmegaatlas.nii'

    cmd = 'bash -lc "fnirt --ref={0} --in={1} --aff={2} --cout={3}"'.format(out_registered, atlas_brain, atlas_registered_flirt_mat, warp_name)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    # apply fnirt warp on atlas template:
    atlas_registered = os.path.join(subj_folder+ 'rr' + atlas_brain.split(sep="\\")[-1])
    cmd = 'bash -lc "applywarp --ref={0} --in={1} --out={2} --warp={3} --interp={4}"'.format(out_registered, atlas_brain, atlas_registered, warp_name, 'nn')
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    # apply fnirt warp on atlas labels:
    atlas_labels_registered = os.path.join(subj_folder+ 'r' + atlas_label.split(sep="\\")[-1])
    cmd = 'bash -lc "applywarp --ref={0} --in={1} --out={2} --warp={3} --interp={4}"'.format(out_registered, atlas_label, atlas_labels_registered, warp_name, 'nn')
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    ## FAST segmentation:
    options = r'-t 1 -n 3 -H 0.1 -I 4 -l 10.0 -o'
    cmd = 'bash -lc "fast {0} {1} {2}"'.format(options, out_registered, out_registered)
    cmd = cmd.replace(os.sep,'/')
    os.system(cmd)

    print('Finished file prep for ' +subj_name[:-1])
