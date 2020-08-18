
from advanced_interactive_viz import *

main_folder = subj_folder
group = 'A'
i = 4
s = all_subj_folders[i]
n = all_subj_names[i]
mean_value=0
bundle_name = 'C_L_mct001rt20'

img_name = rf'\{bundle_name}_{group}.png'
slices = [False, True, False]  # slices[0]-axial, slices[1]-saggital, slices[2]-coronal
file_list = os.listdir(main_folder + s)

for file in file_list:
    if r'_brain.nii' in file and file.startswith('r') and 'MPRAGE' in file:
        slice_file = nib.load(pjoin(main_folder + s, file))
        break

bundlei = AdvanceInteractive(main_folder, slice_file, s, n, img_name, bundle_name, slices)
bundlei.load_bund()
bundlei.load_vols()
num_vols = len(bundlei.vols)
bundlei.vols = [mean_value]*num_vols

bundlei.show_bundle_slices()