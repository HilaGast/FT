from dipy.io.streamline import load_trk, save_trk
from all_subj import *
from dipy.tracking.streamline import values_from_volume
import nibabel as nib
from dipy.viz import window, actor
from dipy.tracking.streamline import transform_streamlines
import numpy as np
from weighted_tracts import load_weight_by_img, save_ft
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points


subj = all_subj_folders
names = all_subj_names
s=subj[10]
n=names[10]
folder_name = subj_folder + s
nii_file = rf'{subj_folder}{s}\diff_corrected_b2000_masked_T2.nii'
hardi_img = nib.load(nii_file)
affine = hardi_img.affine
tract_file = load_trk(rf'{subj_folder}{s}\streamlines\Tracts_CC_ARC.trk',"same")
downsamp = 2
affine1, dimensions1, voxel_sizes1, voxel_order1 = tract_file.space_attributes

str1 = tract_file.streamlines
str1 = set_number_of_points(str1,30)
#str3 = load_tractogram(r'F:\Hila\Ax3D_Pack\V6\v7calibration\YA_lab_Yaniv_002398_20210301_1520\streamlines\Tracts_CC.vtk',nii_file,bbox_valid_check=False)
#stream = list(str1)
#str1 = tract_file
#str1._affine = affine
weight_by='rrdiff_corrected_3_2_AxPasi7'
#weight_by = rf'rr{n[1:]}_ADD_along_streamlines'

weight_by_data, affine2 = load_weight_by_img(folder_name, weight_by)

pfr_data = load_weight_by_img(folder_name,'rrdiff_corrected_3_2_AxFr7')[0]
pfr_per_tract = values_from_volume(pfr_data, str1, affine=affine1)

vol_per_tract = values_from_volume(weight_by_data, str1, affine=affine1)

vol_vec = weight_by_data.flatten()
q = np.quantile(vol_vec[vol_vec > 0], 0.95)
mean_vol_per_tract = []
for s in vol_per_tract:
    s = np.asanyarray(s)

    non_out = [s < q]
    mean_v = np.nanmedian(s[non_out])
    mean_vol_per_tract.append(mean_v)
    #mean_vol_per_tract.append(np.nanmedian(s))



hue = [0.25,-0.05]
saturation = [0,1]
scale = [3,12]

if downsamp != 1:
    mean_vol_per_tract= mean_vol_per_tract[::downsamp]
    str1 = str1[::downsamp]

mean_pasi_weighted_img = f'{folder_name}{os.sep}streamlines{os.sep}CC_3-12_Exp_DTI_PreReg_1_ds2_tube0p3.png'

lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                        saturation_range=saturation, scale_range=scale)

#str2 = transform_streamlines(str1, np.linalg.inv(affine2))
streamlines_actor = actor.streamtube(str1, mean_vol_per_tract, linewidth=0.3, lookup_colormap=lut_cmap)
bar = actor.scalar_bar(lut_cmap)
r = window.Scene()
r.add(streamlines_actor)
r.add(bar)
#r.SetBackground(*window.colors.white)

window.show(r)
r.set_camera(r.camera_info())
window.record(r, out_path=mean_pasi_weighted_img, size=(800, 800))


