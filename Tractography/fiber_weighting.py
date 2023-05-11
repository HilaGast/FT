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


def weight_streamlines_by_cm(streamlines, affine, labels, cm, cm_lookup):
    from dipy.tracking import utils

    cm = cm[cm_lookup,:]
    cm = cm[:,cm_lookup]

    m, grouping = utils.connectivity_matrix(streamlines, affine, labels, return_mapping=True, mapping_as_streamlines=True)
    s_list = []
    vec_vols = []
    for nodes in grouping.keys():
        if nodes[0] == 0 or nodes[1] == 0:
            continue
        for s in grouping[nodes]:
            s_list.append(s)
            vec_vols.append(cm[nodes[0]-1,nodes[1]-1])

    return s_list, vec_vols