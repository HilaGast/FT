from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines
from weighted_tracts import *

subj = all_subj_folders
names = all_subj_names

for s, n in zip(subj[27::], names[27::]):
    folder_name = subj_folder + s
    dir_name = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)

    tract_path = f'{dir_name}{n}_wholebrain_4d_labmask.trk'
    streamlines = load_ft(tract_path, nii_file)
    file_list = os.listdir(folder_name)
    for file in file_list:
        if 'cc_mask' in file and file.endswith('.nii'):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            cc_mask_mat = mask_img.get_fdata()
            mask_include = cc_mask_mat == 1
            break

    masked_streamlines = utils.target(streamlines, affine, mask_include)
    #masked_streamlines = utils.target(masked_streamlines, affine, mask_exclude, include=False)
    masked_streamlines = Streamlines(masked_streamlines)
    weighting_streamlines(folder_name, streamlines, bvec_file, show=True, weight_by='2_2_AxPasi7',
                          scale=[3, 10], hue=[0.25, -0.05], saturation=[0.1, 1.0], fig_type='cc')

save_ft(folder_name, n, masked_streamlines, nii_file, file_name='_' + mask_type + '.trk')
