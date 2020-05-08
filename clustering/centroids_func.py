import numpy as np
from FT.all_subj import *
from dipy.io.streamline import load_trk
from FT.weighted_tracts import load_dwi_files, nodes_labels_mega, nodes_by_index_mega
from dipy.tracking import utils
from dipy.viz import window, actor


def find_centroids(streamlines,th,max_clus, folder_name='', vol_weighted = True, weight_by = '1.5_2_AxPasi5'):
    from dipy.segment.clustering import QuickBundles
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

    else:
        return centroids, s_list


def weight_clus_by_vol_img(weight_by,s_list, folder_name, bvec_file):
    from FT.weighted_tracts import weighting_streamlines

    vec_vols = np.zeros(len(s_list))
    for i,s in enumerate(s_list):
        mean_vol_per_tract = weighting_streamlines(folder_name,s,bvec_file,show=False,weight_by=weight_by)
        vec_vols[i] = np.nanmean(mean_vol_per_tract)

    return vec_vols


if __name__ == '__main__':
    main_folder = subj_folder

    for n,s in zip(all_subj_names[1::],all_subj_folders[1::]):
        folder_name = main_folder+s
        dir_name = folder_name + '\streamlines'
        sft_target = load_trk(f'{dir_name}{n}_wholebrain_3d.trk', "same", bbox_valid_check=False)
        streamlines = sft_target.streamlines
        bvec_file = load_dwi_files(folder_name)[6]

        index_to_text_file = r'C:\Users\hila\data\megaatlas\megaatlas2nii.txt'
        idx = nodes_labels_mega(index_to_text_file)[1]
        lab_labels_index, affine = nodes_by_index_mega(folder_name)
        m, grouping = utils.connectivity_matrix(streamlines, affine, lab_labels_index,
                                                return_mapping=True,
                                                mapping_as_streamlines=True)
        mat_file = f'{folder_name}\weighted_mega_wholebrain_4d_labmask.npy'
        con_mat = np.load(mat_file)
        id = np.argsort(idx)
        con_mat = con_mat[id]
        con_mat = con_mat[:, id]

        vec_vols = []
        s_list = []
        for pair, tracts in grouping.items():
            if pair[0] == 0 or pair[1] == 0:
                continue
            else:
                th = 5
                max_clus = 5
                clus_centroids = find_centroids(streamlines, th, max_clus, folder_name, vol_weighted = False)[0]
                s_list.append(clus_centroids)
                vols = [con_mat[pair[0]-1, pair[1]-1]]*len(clus_centroids)
                vec_vols.append(vols)

        scale = [3, 7]
        hue = [0.25, -0.05]
        saturation = [0.1, 1.0]
        lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                               saturation_range=saturation, scale_range=scale)
        bar = actor.scalar_bar(lut_cmap)
        w_actor = actor.streamtube(s_list, vec_vols, linewidth=0.5, lookup_colormap=lut_cmap)
        r = window.Scene()
        r.add(w_actor)
        r.add(bar)
        window.show(r)
        r.set_camera(r.camera_info())
        window.record(r, out_path=f'{dir_name}\centroids.png', size=(800, 800))



