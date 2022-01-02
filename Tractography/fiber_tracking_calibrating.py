import glob
from Tractography.fiber_tracking import *

main_folder = r'F:\data\tractography comparisons'
subj_folders = glob.glob(f'{main_folder}{os.sep}sub*{os.sep}')

tissue_labels_file_name = "dseg.nii.gz"

for sf in subj_folders:
    ses_fold = sorted(glob(fr'{sf}{os.sep}ses*'))[0]
    subj_main_folder = f'{ses_fold}{os.sep}dwi{os.sep}'
    dat = f'{ses_fold}/dwi/*_dir-FWD_space-orig_desc-preproc_dwi.nii.gz'

    # MSMT Tractography:
    parameters_dict = fiber_tracking_parameters(max_angle= 30,sh_order= 8, seed_density= 5, streamlines_lengths_mm= [50, 1000], step_size= 0.2)
    trk1 = Tractography(subj_main_folder, 'msmt' , 'cmc' , 'wm' , parameters_dict, dat, tissue_labels_file_name)
    trk1.fiber_tracking()

    # CSD Tractography:
    parameters_dict = fiber_tracking_parameters(max_angle=30 ,sh_order= 8, seed_density= 4, streamlines_lengths_mm= [30, 500], step_size= 0.5, fa_th= .18)
    trk2 = Tractography(subj_main_folder, 'csd' , 'fa' , 'wb' , parameters_dict, dat, tissue_labels_file_name)
    trk2.fiber_tracking()




