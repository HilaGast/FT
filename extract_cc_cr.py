from extract_bundles import *
from weighted_tracts import *
subj = all_subj_folders
names = all_subj_names

for s, n in zip(subj[8:9], names[8:9]):
    folder_name = subj_folder + s
    tracts_folder = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
bval_file = bvec_file[:-4:] + 'bval'

bundle_dict={'CC':9,'CCMid':10,'CC_ForcepsMajor':11,'CC_ForcepsMinor':12,'CST_L':25,'CST_R':26,'FPT_L':39,'FPT_R':40,'PPT_L':63,'PPT_R':64, 'OPT_L':58, 'OPT_R':59}
#bundle_dict={'CST_L':25,'CST_R':26,'FPT_L':39,'FPT_R':40,'PPT_L':63,'PPT_R':64, 'OPT_L':58, 'OPT_R':59}

rt = 20
mct = 0.001
main_folder = subj_folder

for b,bnum in bundle_dict.items():
    print(b)
    file_bundle_name = b+r'_mct001rt20'
    bundle_num = bnum
    full_bund_name = f'{n}_{file_bundle_name}'
    if os.path.isdir(tracts_folder) and f'{full_bund_name[1::]}.trk' in os.listdir(tracts_folder):
        print('Moving on!')
        continue
    elif not os.path.exists(f'{tracts_folder}{os.sep}{n}_wholebrain_4d_labmask.trk'):
        print('Moving on!')
        continue
    model, recognized_bundle, bundle_labels = extract_one_bundle(file_bundle_name, bundle_num, 8, rt, mct, main_folder)
    print(f'finished to extract {file_bundle_name} for subj {n}')
