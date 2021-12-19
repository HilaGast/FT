import glob
from weighted_tracts import *

main_folder = r'F:\data\balance'
atlas = 'bna'

for folder_name in glob.glob(main_folder + f'{os.sep}e*{os.sep}*{os.sep}*')[85::]:

    gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name, small_delta=15)
    tract_path = glob.glob(f'{folder_name}{os.sep}streamlines{os.sep}*wholebrain*.trk')[0]
    streamlines = load_ft(tract_path, nii_file)

    weighted_connectivity_matrix_mega(streamlines, folder_name, fig_type=f'wholebrain_4d_labmask_{atlas}_FA',
                                      weight_by='_FA', atlas = 'bna')
    weighted_connectivity_matrix_mega(streamlines, folder_name, fig_type=f'wholebrain_4d_labmask_{atlas}',
                                      weight_by='_AxPasi7', atlas = 'bna')