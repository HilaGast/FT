
import glob
from Tractography.fiber_tracking import *
from Tractography.tractography_vis import show_tracts_simple
from Tractography.fiber_weighting import weight_streamlines


subj_main_folder = f'G:\data\V7\TheBase4Ever\YA_lab_Yaniv_002398_20210301_1520'
diff_file = glob.glob(f'{subj_main_folder}{os.sep}*d15D45APs012a001.nii')[0]
nii_ref = os.path.join(diff_file)
affine = nib.load(nii_ref).affine


tract_file = subj_main_folder+f'{os.sep}streamlines{os.sep}002398_wholebrain_5d_labmask_msmt.trk'
hue = [0.25,-0.05]
saturation = [0.1,1]
scale = [6,10]
fig_type = 'WB_front'
streamlines = load_ft(tract_file, nii_ref)
s_list = [s1 for s1 in streamlines]
mean_vol_per_tract = weight_streamlines(streamlines,subj_main_folder, weight_by='3_2_AxPasi7')
show_tracts_simple(s_list, subj_main_folder, fig_type, down_samp=30, vec_vols=mean_vol_per_tract,hue=hue,saturation=saturation,scale=scale, weighted=True)
