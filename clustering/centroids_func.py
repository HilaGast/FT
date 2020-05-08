import numpy as np
from FT.all_subj import *


def find_centroids(streamlines,th,max_clus, folder_name='', vol_weighted = True, weight_by = '1.5_2_AxPasi5'):
    from dipy.segment.clustering import QuickBundles
    from FT.weighted_tracts import load_dwi_files
    ''' parameters:
        streamlines: ArraySequence of streamlines
        th: float for max dist from a streamline to the centroids.
            Above this value a new centroid is created as long as the number of centroids < max clus.
        max_clus: int of max number of clusters

        :return:
        centroids: a list of streamilnes centroid representation
        s_list: a list of lists. 
            Each inner list is the group of streamlines cooresponds to a single centroid.
    '''
    qb = QuickBundles(th,max_nb_clusters=max_clus)
    qbmap = qb.cluster(streamlines)
    centroids = qbmap.centroids
    s_list = []
    for i in qbmap.clusters:
        s_list.append(list(i))

    if vol_weighted:
        bvec_file = load_dwi_files(folder_name)[6]
        vec_vols = weight_clus_by_vol_img(weight_by,s_list, folder_name, bvec_file)

    return centroids, s_list, vec_vols


def weight_clus_by_vol_img(weight_by,s_list, folder_name, bvec_file):
    from FT.weighted_tracts import weighting_streamlines

    vec_vols = np.zeros(len(s_list))
    for i,s in enumerate(s_list):
        mean_vol_per_tract = weighting_streamlines(folder_name,s,bvec_file,weight_by)
        vec_vols[i] = np.nanmean(mean_vol_per_tract)

    return vec_vols



