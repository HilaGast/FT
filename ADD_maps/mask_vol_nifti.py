from Tractography.files_loading import load_ft, load_nii_file
from Tractography.fiber_weighting import weight_streamlines
import numpy as np
import nibabel as nib
from glob import glob
import os
from dipy.tracking.streamline import set_number_of_points


def vol_map(streamlines, affine, vol_vec, vol_dims):
    from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
    """Calculates the mean volume of the streamlines that pass through each voxel.
    Parameters
    ----------
    streamlines : iterable
        A sequence of streamlines.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline points.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    vol_vec : ndarray, shape (1, N) while N is the number of streamlines
        The volume (ADD for example) according which the code calculates the new image.
    vol_dims : 3 ints
        The shape of the volume to be returned containing the streamlines
        counts
    Returns
    -------
    vol_vox : ndarray, shape=vol_dims
        The mean volume of the streamlines that pass through each voxel.
    Raises
    ------
    IndexError
        When the points of the streamlines lie outside of the return volume.
    Notes
    -----
    A streamline can pass through a voxel even if one of the points of the
    streamline does not lie in the voxel. For example a step from [0,0,0] to
    [0,0,2] passes through [0,0,1]. Consider subsegmenting the streamlines when
    the edges of the voxels are smaller than the steps of the streamlines.
    """
    streamlines = set_number_of_points(streamlines,50)
    lin_T, offset = _mapping_to_voxel(affine)
    counts = np.zeros(vol_dims, 'int')
    vols_sum = np.zeros(vol_dims, 'float64')
    for sl,vl in zip(streamlines,vol_vec):
        inds = _to_voxel_coordinates(sl, lin_T, offset)
        i, j, k = inds.T
        # this takes advantage of the fact that numpy's += operator only
        # acts once even if there are repeats in inds
        counts[i, j, k] += 1
        vols_sum[i, j, k] += vl
    vox_vol = np.true_divide(vols_sum, counts, out=np.zeros_like(vols_sum), where = counts != 0)

    return vox_vol



if __name__ == '__main__':
    # main_feo = r'F:\Hila\balance\eo'
    # main_fec = r'F:\Hila\balance\ec'
    # subj_folder = glob(os.path.join(main_feo,'*/')) +glob(os.path.join(main_fec,'*/'))
    subj_folder = glob(fr'G:\data\V7\HCP{os.sep}*{os.sep}')
    all_subj_folders = []
    for folder_name in subj_folder[::]:
        dir_name = folder_name + 'streamlines'
        vol_file_name = folder_name+f'average_add_map.nii'
        if os.path.exists(vol_file_name):
            continue
        try:
            diff_file = glob(f'{folder_name}data_1st.nii.gz')[0]
        except IndexError:
            try:
                diff_file = glob(f'{folder_name}data_1st.nii')[0]
            except IndexError:
                continue
        data, affine = load_nii_file(diff_file)

        for tfiles in glob(os.path.join(dir_name, '*')):
            #if 'wholebrain_5d_labmask_msmt.trk' in tfiles:
            if 'HCP_tracts.tck' in tfiles:
                tract_path = os.path.join(dir_name, tfiles)
                continue
        streamlines = load_ft(tract_path, diff_file)
        mean_vol_per_tract = weight_streamlines(streamlines, folder_name, weight_by='3_2_AxPasi7')
        vox_vol = vol_map(streamlines, affine, mean_vol_per_tract, data.shape)

        empty_header = nib.Nifti1Header()
        vol_img = nib.Nifti1Image(vox_vol,affine,empty_header)
        nib.save(vol_img,vol_file_name)

        # vox_vol_masked = vox_vol * white_matter
        # masked_vol_img = nib.Nifti1Image(vox_vol_masked, affine, empty_header)
        # masked_vol_file_name = os.path.join(folder_name, 'average_add_map_WMmasked.nii')
        # nib.save(masked_vol_img, masked_vol_file_name)

