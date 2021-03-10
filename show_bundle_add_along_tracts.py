from weighted_tracts import *
from dipy.tracking.streamline import transform_streamlines

fig_type='cc'
weight_by = '1.5_2.5_AxPasi7'
hue=[0.25, -0.05] #hot
#hue = [0.4, 0.7] # blues
saturation = [0.0, 1.0]
scale = [2, 10]
subj = all_subj_folders
names = all_subj_names

for s, n in zip(subj[8:9], names[8:9]):
    folder_name = subj_folder + s
    dir_name = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)

weight_by_data, affine1 = load_weight_by_img(folder_name, weight_by)

#tract_path = folder_name+r'\streamlines'+n+'_'+fig_type+'_4d_labmask_msmt.trk'
tract_path = folder_name+r'\streamlines'+n+'_'+'CC_mct001rt20_msmt.trk'
streamlines = load_ft(tract_path,nii_file)
stream = list(streamlines)

streamlines = transform_streamlines(streamlines, np.linalg.inv(affine1))

show_tracts(hue ,saturation ,scale ,streamlines ,weight_by_data ,folder_name ,fig_type +'_'+weight_by+'_add_a')
show_tracts(hue ,saturation ,scale ,streamlines ,weight_by_data ,folder_name ,fig_type +'_'+weight_by+'_add_b')

