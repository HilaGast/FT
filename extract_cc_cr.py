from extract_bundles import *
from weighted_tracts import *
subj = all_subj_folders
names = all_subj_names

#bundle_dict={'CC':9,'CCMid':10,'CC_ForcepsMajor':11,'CC_ForcepsMinor':12,'CST_L':25,'CST_R':26,'FPT_L':39,'FPT_R':40,'PPT_L':63,'PPT_R':64, 'OPT_L':58, 'OPT_R':59,'AF_L':1,'AF_R':2,'C_L':33,'C_R':34}
bundle_dict={'OPT_L':58, 'OPT_R':59,'CC':9,'CST_L':25,'CST_R':26,'FPT_L':39,'FPT_R':40,'PPT_L':63,'PPT_R':64 }
#bundle_dict={'CST_L':25,'CST_R':26,'FPT_L':39,'FPT_R':40,'PPT_L':63,'PPT_R':64, 'OPT_L':58, 'OPT_R':59,'AF_L':1,'AF_R':2,'C_L':33,'C_R':34}
#bundle_dict={'AF_L':1,'AF_R':2,'C_L':33,'C_R':34}
rt = 20
mct = 0.01

for (i,s), n in zip(enumerate(subj[10:11]), names[10:11]):
    print(n)
    folder_name = subj_folder + s
    tracts_folder = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
    bval_file = bvec_file[:-4:] + 'bval'

    main_folder = subj_folder

    wb_name = '_wholebrain_3d_labmask_sh6_cmc_pft'
    if not os.path.exists(f'{tracts_folder}{os.sep}{n}{wb_name}.trk'):
        print('Moving on!')
        continue

    moved, target = transform_bundles(folder_name, n, wb_tracts_type=wb_name)
    for b,bnum in bundle_dict.items():
        print(b)
        file_bundle_name = b+r'_mct001rt20_3d_cmc_pft'
        #file_bundle_name = b+r'_mct001rt20_msmt_5d'

        bundle_num = bnum
        full_bund_name = f'{n}_{file_bundle_name}'
        if os.path.isdir(tracts_folder) and f'{full_bund_name[1::]}.trk' in os.listdir(tracts_folder):
            print('Moving on!')
            continue
        else:
            model, recognized_bundle, bundle_labels = extract_one_bundle(moved, target, file_bundle_name, bundle_num, n, folder_name, rt, mct)
            print(f'finished to extract {file_bundle_name} for subj {n}')
