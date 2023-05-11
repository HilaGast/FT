from Tractography.connectivity_matrices import ConMat, WeightConMat
from Tractography.fiber_tracking import fiber_tracking_parameters, Tractography
import glob, os
from Tractography.files_loading import load_ft

main_fol = 'F:\Hila\TDI\TheBase4Ever'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}[0-9]*{os.sep}')
#all_atlas = ['bnacor', 'yeo7_200']
tissue_labels_file_name = 'rMPRAGE_brain_seg.nii'
for atlas in all_atlas:
    for subj in all_subj_fol:
        dat = subj+'diff_corrected.nii'
        if not os.path.exists(subj + f'streamlines{os.sep}wb_csd_fa.tck'):
            parameters_dict = fiber_tracking_parameters(max_angle=30, sh_order=6, seed_density=4,
                                                streamlines_lengths_mm=[50, 500], step_size=1, fa_th=.18)
            tracts = Tractography(subj, 'csd', 'fa', 'wb', parameters_dict, dat,
                        tissue_labels_file_name=tissue_labels_file_name)
            tracts.fiber_tracking()
            streamlines = tracts.streamlines