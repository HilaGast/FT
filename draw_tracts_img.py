from FT.weighted_tracts import weighting_streamlines,load_ft, load_dwi_files
from FT.all_subj import all_subj_names,all_subj_folders

subj = all_subj_folders
names = all_subj_names
masks = ['cc_cortex_cleaned','wholebrain']
weight_by='1.5_2_AxPasi5'

for n,s in zip(names,subj):
    folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s
    dir_name = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)

    for m in masks:
        tract_path = dir_name + n + '_' + m +'.trk'
        streamlines = load_ft(tract_path,nii_file)
        fig_name = m + '_example'
        weighting_streamlines(folder_name, streamlines, bvec_file, weight_by='1.5_2_AxPasi5', hue=[0.0, 1.0],
                              saturation=[0.0, 1.0], scale=[2, 7], fig_type=fig_name)

