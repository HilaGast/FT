from Tractography.connectivity_matrices import ConMat, WeightConMat
from Tractography.fiber_tracking import fiber_tracking_parameters, Tractography
import glob, os

main_fol = 'F:\Hila\siemens'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')

experiments = ['D31d18','D45d13','D60d11']
tissue_labels_file_name = 'mprage_reg_seg.nii'

for experiment in experiments[:2]:
    for fol in all_subj_fol:
        exp_fol = f'{fol}{experiment}{os.sep}'
        dat = f'{exp_fol}diff_corrected_{experiment}.nii'

        if not os.path.exists(exp_fol + f'streamlines{os.sep}wb_csd_fa.tck'):
            parameters_dict = fiber_tracking_parameters(max_angle=30, sh_order=6, seed_density=4,
                                                streamlines_lengths_mm=[50, 500], step_size=1, fa_th=.18)
            tracts = Tractography(exp_fol, 'csd', 'fa', 'wb', parameters_dict, dat,
                        tissue_labels_file_name=tissue_labels_file_name)
            tracts.fiber_tracking()

        if not os.path.exists(f'{exp_fol}cm{os.sep}add_bna_cm_ord.npy'):
            cm = ConMat(atlas='bna', diff_file = dat, subj_folder = exp_fol, tract_name='wb_csd_fa.tck')
            cm.save_cm(fig_name='num_bna', mat_type='cm_ord')

            cm = WeightConMat(weight_by='ADD', atlas='bna', diff_file=dat, subj_folder=exp_fol,
                              tract_name='wb_csd_fa.tck')
            cm.save_cm(fig_name='add_bna', mat_type='cm_ord')
