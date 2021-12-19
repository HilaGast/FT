from dipy.data.fetcher import get_bundle_atlas_hcp842
import numpy as np
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr
from fury import actor, window
from dipy.io.streamline import load_trk
from os.path import join as pjoin
import os
from all_subj import all_subj_folders, all_subj_names, subj_folder
from weighted_tracts import save_ft,load_dwi_files
from remove_cci_outliers import remove_cci_outliers
from dipy.tracking.streamline import set_number_of_points
import glob

def find_home():
    if 'DIPY_HOME' in os.environ:
        dipy_home = os.environ['DIPY_HOME']
    else:
        dipy_home = pjoin(os.path.expanduser('~'), '.dipy')

    return dipy_home


def show_atlas_target_graph(atlas,target,out_path,interactive=True):
    ren = window.Scene()
    ren.SetBackground(1, 1, 1)
    ren.add(actor.line(atlas, colors=(1, 0, 1)))  # Magenta
    ren.add(actor.line(target, colors=(1, 1, 0)))  # Yellow
    #window.record(ren, out_path=out_path, size=(600, 600))
    if interactive:
        window.show(ren)


def find_bundle(dipy_home,moved,bundle_num, rt=50,mct=0.1):
    bundle_folder = dipy_home+r'\bundle_atlas_hcp842\Atlas_80_Bundles\bundles'
    bundles = os.listdir(bundle_folder)
    model_file = pjoin(dipy_home, 'bundle_atlas_hcp842',
                       'Atlas_80_Bundles',
                       'bundles',
                       bundles[bundle_num])

    sft_model = load_trk(model_file, "same", bbox_valid_check=False)
    model = sft_model.streamlines
    moved = set_number_of_points(moved, 20)
    rb = RecoBundles(moved, verbose=True, rng=np.random.RandomState(2001),nb_pts=20)
    #model = set_number_of_points(model,20)
    recognized_bundle, bundle_labels = rb.recognize(model_bundle=model,
                                                  model_clust_thr=mct,
                                                  reduction_thr=rt,
                                                  reduction_distance='mam',
                                                  slr=True,
                                                  slr_metric='asymmetric',
                                                  pruning_distance='mam')
    return recognized_bundle,bundle_labels, model


def transform_bundles(wb_tracts_name):
    atlas_file, all_bundles_files = get_bundle_atlas_hcp842()
    sft_atlas = load_trk(atlas_file, "same", bbox_valid_check=False)
    atlas = sft_atlas.streamlines
    sft_target = load_trk(wb_tracts_name, "same", bbox_valid_check=False)

    target = sft_target.streamlines
    #show_atlas_target_graph(atlas, target,out_path=folder_name+r'\try_atlas_target',interactive=True)
    atlas = set_number_of_points(atlas,20)
    target = set_number_of_points(target,20)
    moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
    atlas, target, x0='affine', verbose=True, progressive=True,
    rng=np.random.RandomState(1984))
    #np.save("slf_L_transform.npy", transform)
    #show_atlas_target_graph(atlas, moved,out_path=r'',interactive=True)

    return moved, target


def extract_one_bundle(moved, target, full_bund_name, bundle_num, subj, folder_name, rt, mct):
    dipy_home = find_home()
    #moved = set_number_of_points(moved, 20)
    recognized_bundle,bundle_labels, model = find_bundle(dipy_home,moved,bundle_num, rt, mct)
    nii_file = load_dwi_files(folder_name)[5]
    bundle = target[bundle_labels]

    if len(bundle)<20:
        model=[]
        recognized_bundle=[]
        bundle_labels=[]
        print(f"Couldn't find {full_bund_name} for {subj}")
    else:
        keep_s,keep_i = remove_cci_outliers(bundle)
        new_s = []
        new_s += [bundle[s1] for s1 in keep_i]
        save_ft(folder_name, '', new_s, nii_file, file_name=rf'\{full_bund_name}.trk')

    return model,recognized_bundle,bundle_labels



def show_model_reco_bundles(model,recognized_bundle,folder_name,file_bundle_name,interactive=True):

    ren = window.Scene()
    ren.SetBackground(1, 1, 1)
    ren.add(actor.line(model, colors=(.1, .7, .26))) #green
    ren.add(actor.line(recognized_bundle, colors=(.1, .1, 6))) #blue

    if interactive:
       window.show(ren)

    ren.set_camera(ren.camera_info())
    window.record(ren, out_path=pjoin(folder_name,file_bundle_name)+'.png', size=(600, 600))


if __name__ == '__main__':
    bundle_dict = {1:'AF_L',2:'AF_R',68:'SLF_L',69:'SLF_R'}
    main_folder = r'C:\Users\Admin\Desktop\Language'
    rt=20
    mct=0.01
    for subj in glob.glob(f'{main_folder}{os.sep}*{os.sep}'):
        s = str.split(subj,os.sep)[-1]
        tracts_folder = f'{subj}\streamlines'
        wb_bundle_name = glob.glob(tracts_folder + '*/*_wholebrain_5d_labmask_msmt.trk')

        if not wb_bundle_name:
            print('Moving on!')
            continue

        wb_bundle_name = wb_bundle_name[0]

        moved, target = transform_bundles(wb_bundle_name)

        for bnum, b in bundle_dict.items():
            print(b)
            file_bundle_name = b + r'_mct01rt20_5d'

            full_bund_name = f'{s}_{file_bundle_name}'
            if os.path.isdir(tracts_folder) and f'{full_bund_name}.trk' in os.listdir(tracts_folder):
                print('Moving on!')
                continue
            else:
                model, recognized_bundle, bundle_labels = extract_one_bundle(moved, target, full_bund_name,
                                                                             bnum, subj, subj, rt, mct)
                print(f'finished to extract {file_bundle_name} for subj {s}')





