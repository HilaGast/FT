from extract_bundles import *
from weighted_tracts import *
from blinded.wholebrain_tract import load_dwi_files_blinds

subj_folder = r'C:\Users\HilaG\Desktop\blind dti\blind'
all_folders = os.listdir(subj_folder)
dipy_home = find_home()
atlas_file, all_bundles_files = get_bundle_atlas_hcp842()
sft_atlas = load_trk(atlas_file, "same", bbox_valid_check=False)
atlas = sft_atlas.streamlines
bundle_dict={'OR_L':60,'OR_R':61}
rt = 20
mct = 0.001
for b,bnum in bundle_dict.items():

    bundle_num = bnum
    file_bundle_name = b+r'_mct001rt20_labmask_5d'

    for s in all_folders[::]:
        name = s
        name = '/' + name
        n = name.replace('/', os.sep)
        folder_name = subj_folder + n
        dir_name = folder_name + '\streamlines'

        sft_target = load_trk(dir_name + n + r'_wholebrain_5d_labelmask.trk', "same", bbox_valid_check=False)
        target = sft_target.streamlines

        moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
            atlas, target, x0='affine', verbose=True, progressive=True,
            rng=np.random.RandomState(1984))

        recognized_bundle, bundle_labels, model = find_bundle(dipy_home, moved, bundle_num, rt, mct)
        nii_file = load_dwi_files_blinds(folder_name)[5]
        bundle = target[bundle_labels]

        if len(bundle) < 20:
            model = []
            recognized_bundle = []
            bundle_labels = []
            print(f"Couldn't find {file_bundle_name} for {n}")
        else:
            keep_s, keep_i = remove_cci_outliers(bundle)
            new_s = []
            new_s += [bundle[s1] for s1 in keep_i]
            save_ft(folder_name, n, new_s, nii_file, file_name='_' + file_bundle_name + '.trk')

        print(f'finished to extract {file_bundle_name} for subj {n}')




