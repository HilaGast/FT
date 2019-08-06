
import os

subj_name = r'ShEl_subj2/'
mprage_file_name = r'20190121_122555T1wMPRAGERLs002a1001.nii'
first_charmed_file_name = r'f20190121_101217ep2dd155D60MB3APs005a001_01.nii'
atlas_template = r'C:\Users\Admin\my_scripts\aal\origin\AAL_highres_template.nii'
atlas_template = atlas_template.replace('C:', '/mnt/c')
atlas_label = r'C:\Users\Admin\my_scripts\aal\origin\AAL_highres_atlas.nii'
atlas_label = atlas_label.replace('C:', '/mnt/c')

main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5/'
main_folder = main_folder.replace('C:', '/mnt/c')
subj_folder = main_folder + subj_name

## Registration from MPRAGE to 1st CHARMED scan using inverse matrix of CHARMED to MPRAGE registration:
# From CHARMED to MPRAGE:
subj_mprage = subj_folder + mprage_file_name
subj_first_charmed = subj_folder + first_charmed_file_name
out_registered = subj_folder + 'r' + first_charmed_file_name
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
out_registered = subj_folder + 'r' + mprage_file_name
cmd = 'bash -lc "flirt -in {0} -ref {1} -out {2} -applyxfm -init {3}"'.format(subj_mprage, subj_first_charmed, out_registered, inv_mat)
cmd = cmd.replace(os.sep,'/')
os.system(cmd)

## BET for registered MPRAGE and mni template:
# BET for registered MPRAGE:
out_brain = out_registered[:-4]+'_brain'

cmd = 'bash -lc "bet {0} {1} {2} {3}"'.format(out_registered[:-4], out_brain,'-f 0.45','-g 0.2')
cmd = cmd.replace(os.sep,'/')
os.system(cmd)

# BET for mni template:

'''
# if not performed before, run:
atlas_brain = atlas_template[:-4] + '_brain'
cmd = 'bash -lc "bet {0} {1} {2} {3}"'.format(atlas_template[:-4], atlas_brain,'-f 0.45','-g -0.1')
cmd = cmd.replace(os.sep,'/')
os.system(cmd)
'''
## Registration from MNI to regisered MPRAGE:

# flirt for MNI to registered MPRAGE for primary guess:
options = r'-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear'
out_brain = out_brain + '.nii'
atlas_brain = atlas_template[:-4] + '_brain.nii'

atlas_registered_flirt = os.path.join(subj_folder+ 'r' + atlas_brain.split(sep="\\")[-1])
atlas_registered_flirt_mat = atlas_registered_flirt[:-4] + '.mat'

cmd = 'bash -lc "flirt -ref {0} -in {1} -out {2} -omat {3} {4}"'.format(out_brain, atlas_brain, atlas_registered_flirt, atlas_registered_flirt_mat, options)
cmd = cmd.replace(os.sep,'/')
os.system(cmd)

# fnirt for MNI based on flirt results:
warp_name = subj_folder + 'atlas2subj.nii'
cmd = 'bash -lc "fnirt --ref={0} --in={1} --aff={2} --cout={3}"'.format(out_brain, atlas_brain, atlas_registered_flirt_mat, warp_name)
cmd = cmd.replace(os.sep,'/')
os.system(cmd)

# apply fnirt warp on atlas template:
atlas_registered = os.path.join(subj_folder+ 'rr' + atlas_brain.split(sep="\\")[-1])
cmd = 'bash -lc "applywarp --ref={0} --in={1} --out={2} --warp={3}"'.format(out_brain, atlas_brain, atlas_registered, warp_name)
cmd = cmd.replace(os.sep,'/')
os.system(cmd)

# apply fnirt warp on atlas labels:
atlas_labels_registered = os.path.join(subj_folder+ 'r' + atlas_label.split(sep="\\")[-1])
cmd = 'bash -lc "applywarp --ref={0} --in={1} --out={2} --warp={3}"'.format(out_brain, atlas_label, atlas_labels_registered, warp_name)
cmd = cmd.replace(os.sep,'/')
os.system(cmd)

## FAST segmentation:
options = r'-t 1 -n 3 -H 0.1 -I 4 -l 10.0 -o'
cmd = 'bash -lc "fast {0} {1} {2}"'.format(options, out_brain, out_brain)
cmd = cmd.replace(os.sep,'/')
os.system(cmd)

print('Finished file prep for ' +subj_name[:-1])
