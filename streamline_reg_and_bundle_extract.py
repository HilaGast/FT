from dipy.data.fetcher import get_two_hcp842_bundles

from dipy.data.fetcher import (fetch_bundle_atlas_hcp842,
                               get_bundle_atlas_hcp842)
import numpy as np
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr
from fury import actor, window
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_trk, save_trk
from dipy.io.utils import create_tractogram_header


target_file, target_folder = fetch_target_tractogram_hcp()
atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()

atlas_file, all_bundles_files = get_bundle_atlas_hcp842()
target_file = get_target_tractogram_hcp()

sft_atlas = load_trk(atlas_file, "same", bbox_valid_check=False)
atlas = sft_atlas.streamlines
atlas_header = create_tractogram_header(atlas_file,
                                        *sft_atlas.space_attribute)

sft_target = load_trk(target_file, "same", bbox_valid_check=False)
target = sft_target.streamlines
target_header = create_tractogram_header(atlas_file,
                                         *sft_atlas.space_attribute)

moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
    atlas, target, x0='affine', verbose=True, progressive=True)

model_af_l_file, model_cst_l_file = get_two_hcp842_bundles()

sft_af_l = load_trk(model_af_l_file, "same", bbox_valid_check=False)
model_af_l = sft_af_l.streamlines

rb = RecoBundles(moved, verbose=True)

recognized_af_l, af_l_labels = rb.recognize(model_bundle=model_af_l,
                                            model_clust_thr=5.,
                                            reduction_thr=10,
                                            reduction_distance='mam',
                                            slr=True,
                                            slr_metric='asymmetric',
                                            pruning_distance='mam')

reco_af_l = StatefulTractogram(target[af_l_labels], target_header,
                               Space.RASMM)
save_trk(reco_af_l, "AF_L.trk", bbox_valid_check=False)

sft_cst_l = load_trk(model_cst_l_file, "same", bbox_valid_check=False)
model_cst_l = sft_cst_l.streamlines

recognized_cst_l, cst_l_labels = rb.recognize(model_bundle=model_cst_l,
                                              model_clust_thr=5.,
                                              reduction_thr=10,
                                              reduction_distance='mam',
                                              slr=True,
                                              slr_metric='asymmetric',
                                              pruning_distance='mam')

reco_cst_l = StatefulTractogram(target[cst_l_labels], target_header,
                                Space.RASMM)
save_trk(reco_cst_l, "CST_L.trk", bbox_valid_check=False)