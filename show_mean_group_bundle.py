
from advanced_interactive_viz import *

#main_folder = subj_folder
group = 'D'
#i = 4
#s = all_subj_folders[i]
#n = all_subj_names[i]
main_folder = r'C:\Users\Admin\Desktop\אלפא-ליהיא'
s = r'\YA_lab_Yaniv_001406_20200213_1255'
n= r'\001406'
mean_value=5.64
bundle_name = 'F_L_R_mct001rt20'

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

bundlei.scale=[5,6]
bundlei.show_bundle_slices()