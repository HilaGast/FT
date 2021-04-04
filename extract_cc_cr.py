from extract_bundles import *
from weighted_tracts import *
subj = all_subj_folders
names = all_subj_names

for s, n in zip(subj[9:10], names[9:10]):
    folder_name = subj_folder + s
    tracts_folder = folder_name + '\streamlines'
    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
bval_file = bvec_file[:-4:] + 'bval'

bundle_dict={'CC':9,'CCMid':10,'CC_ForcepsMajor':11,'CC_ForcepsMinor':12,'CST_L':25,'CST_R':26,'FPT_L':39,'FPT_R':40,'PPT_L':63,'PPT_R':64, 'OPT_L':58, 'OPT_R':59,'AF_L':1,'AF_R':2,'C_L':33,'C_R':34}
#bundle_dict={'CST_L':25,'CST_R':26,'FPT_L':39,'FPT_R':40,'PPT_L':63,'PPT_R':64, 'OPT_L':58, 'OPT_R':59,'AF_L':1,'AF_R':2,'C_L':33,'C_R':34}
#bundle_dict={'AF_L':1,'AF_R':2,'C_L':33,'C_R':34}
rt = 20
mct = 0.001
main_folder = subj_folder

for b,bnum in bundle_dict.items():
    print(b)
    file_bundle_name = b+r'_mct001rt20_msmt_5d'
    bundle_num = bnum
    full_bund_name = f'{n}_{file_bundle_name}'
    if os.path.isdir(tracts_folder) and f'{full_bund_name[1::]}.trk' in os.listdir(tracts_folder):
        print('Moving on!')
        continue
    elif not os.path.exists(f'{tracts_folder}{os.sep}{n}_wholebrain_5d_labmask_msmt.trk'):
        print('Moving on!')
        continue
    else:
        model, recognized_bundle, bundle_labels = extract_one_bundle(file_bundle_name, bundle_num, 9, rt, mct, main_folder)
        print(f'finished to extract {file_bundle_name} for subj {n}')
