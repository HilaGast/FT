from dipy.data.fetcher import get_bundle_atlas_hcp842
import numpy as np
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr
from fury import actor, window
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_trk, save_trk
from os.path import join as pjoin
import os
from FT.all_subj import all_subj_folders, all_subj_names
from FT.weighted_tracts import save_ft,load_dwi_files

def find_home():
    if 'DIPY_HOME' in os.environ:
        dipy_home = os.environ['DIPY_HOME']
    else:
        dipy_home = pjoin(os.path.expanduser('~'), '.dipy')

    return dipy_home


def show_atlas_target_graph(atlas,target,out_path,interactive=True):
    ren = window.Renderer()
    ren.SetBackground(1, 1, 1)
    ren.add(actor.line(atlas, colors=(1, 0, 1)))  # Magenta
    ren.add(actor.line(target, colors=(1, 1, 0)))  # Yellow
    window.record(ren, out_path=out_path, size=(600, 600))
    if interactive:
        window.show(ren)


def find_bundle(dipy_home,moved,bundle_num, rt=50,mct=0.1):
    bundle_folder = r'C:\Users\Admin\.dipy\bundle_atlas_hcp842\Atlas_80_Bundles\bundles'
    bundles = os.listdir(bundle_folder)
    model_file = pjoin(dipy_home, 'bundle_atlas_hcp842',
                       'Atlas_80_Bundles',
                       'bundles',
                       bundles[bundle_num])

    sft_model = load_trk(model_file, "same", bbox_valid_check=False)
    model = sft_model.streamlines

    rb = RecoBundles(moved, verbose=True, rng=np.random.RandomState(2001))

    recognized_bundle, bundle_labels = rb.recognize(model_bundle=model,
                                                  model_clust_thr=mct,
                                                  reduction_thr=rt,
                                                  reduction_distance='mam',
                                                  slr=True,
                                                  slr_metric='asymmetric',
                                                  pruning_distance='mam')
    return recognized_bundle,bundle_labels, model


if __name__ == '__main__':
    file_bundle_name = r'C_L_recognized_bundle'
    bundle_num = 33
    dipy_home = find_home()
    atlas_file, all_bundles_files = get_bundle_atlas_hcp842()
    sft_atlas = load_trk(atlas_file, "same", bbox_valid_check=False)
    atlas = sft_atlas.streamlines
    main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'
    folder_name = main_folder+all_subj_folders[0]
    n = all_subj_names[0]

    sft_target = load_trk(folder_name + r'\streamlines'+n+r'_wholebrain_3d.trk', "same", bbox_valid_check=False)
    target = sft_target.streamlines

    #show_atlas_target_graph(atlas, target,outpath=r'',interactive=True)
    moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
    atlas, target, x0='affine', verbose=True, progressive=True,
    rng=np.random.RandomState(1984))


    #np.save("slr_transform.npy", transform)
    #show_atlas_target_graph(atlas, moved,outpath=r'',interactive=True)
    rt=50
    mct=0.1
    recognized_bundle,bundle_labels, model = find_bundle(dipy_home,moved,bundle_num, rt, mct)
    nii_file = load_dwi_files(folder_name)[5]
    save_ft(folder_name, n, target[bundle_labels], nii_file, file_name='_'+file_bundle_name+'.trk')
    interactive = True
    ren = window.Renderer()
    ren.SetBackground(1, 1, 1)
    ren.add(actor.line(model, colors=(.1, .7, .26))) #green
    ren.add(actor.line(recognized_bundle, colors=(.1, .1, 6))) #blue

    if interactive:
       window.show(ren)

    ren.set_camera(ren.camera_info())
    window.record(ren, out_path=pjoin(folder_name,file_bundle_name)+'.png', size=(600, 600))

