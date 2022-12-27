from Tractography.files_loading import load_weight_by_img
import numpy as np


def weight_streamlines(streamlines, folder_name, weight_by='3_2_AxPasi7'):
    from dipy.tracking.streamline import values_from_volume

    weight_by_data, affine = load_weight_by_img(folder_name,weight_by)
    stream = list(streamlines)
    vol_per_tract = values_from_volume(weight_by_data, stream, affine=affine)
    vol_vec = weight_by_data.flatten()
    q = np.quantile(vol_vec[vol_vec>0], 0.95)
    mean_vol_per_tract = []
    for s in zip(vol_per_tract):
        s = np.asanyarray(s)
        non_out = [s < q]
        mean_vol_per_tract.append(np.nanmean(s[non_out]))

    return mean_vol_per_tract