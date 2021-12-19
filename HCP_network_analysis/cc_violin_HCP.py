from cc_analysis.cc_boxplot import create_cc_vioplot, detect_and_remove_outliers
import glob, os
import numpy as np
import pandas as pd
import nibabel as nib


def load_tck(tck_file_name, nii_file_name):
    from dipy.io.streamline import load_tck, Space

    streams = load_tck(tck_file_name, nii_file_name, Space.RASMM)
    streamlines = streams.get_streamlines_copy()

    return streamlines

def load_cc_mask_parts(dir_name):

    mask_genu = nib.load(os.path.join(dir_name,'mask_genu.nii')).get_fdata()
    mask_abody = nib.load(os.path.join(dir_name,'mask_abody.nii')).get_fdata()
    mask_mbody = nib.load(os.path.join(dir_name,'mask_mbody.nii')).get_fdata()
    mask_pbody = nib.load(os.path.join(dir_name,'mask_pbody.nii')).get_fdata()
    mask_splenium = nib.load(os.path.join(dir_name,'mask_splenium.nii')).get_fdata()

    return mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium

def calc_cc_part_val(streamlines, mask, affine, calc_type='mean'):
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines
    from weighted_tracts import weighting_streamlines

    masked_streamlines = utils.target(streamlines, affine, mask)

    masked_streamlines = Streamlines(masked_streamlines)

    weights = weighting_streamlines(dir_name, masked_streamlines, affine, show=False, weight_by='3_2_AxPasi7')

    if 'mean' in calc_type:
        val = np.nanmean(weights)
    elif 'median' in calc_type:
        val = np.nanmedian(weights)

    return val


if __name__ == '__main__':

    shortlist = glob.glob(f'F:\data\V7\HCP\*{os.sep}')
    val_mat = []
    for sl in shortlist:
        dir_name = sl[:-1]
        subj_number = sl.split(os.sep)[-2]
        if not os.path.isfile(sl + 'HCP_tracts.tck'):
            continue
        else:
            tck_file_name = sl + 'HCP_tracts.tck'
            nii_file_name = sl + 'data.nii'
            affine = nib.load(nii_file_name).affine
            streamlines = load_tck(tck_file_name, nii_file_name)
            mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_cc_mask_parts(dir_name)
            val_vec = []

            for mask in [mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium]:
                val = calc_cc_part_val(streamlines, mask, affine, calc_type='median')
                val_vec.append(val)
            val_mat.append(val_vec)

    val_mat = np.asarray(val_mat)
    cc_parts_table = pd.DataFrame(val_mat, columns=['Genu', 'Anterior Body', 'Mid Body', 'Posterior Body', 'Splenium'])
    cc_parts_table, num_lo = detect_and_remove_outliers(cc_parts_table)
    print(cc_parts_table)
    print(f'Removed {num_lo} outliers')
    create_cc_vioplot(cc_parts_table)

    from scipy.stats import f_oneway
    print(f_oneway(cc_parts_table['Genu'], cc_parts_table['Anterior Body'], cc_parts_table['Mid Body'],
             cc_parts_table['Posterior Body'], cc_parts_table['Splenium']))