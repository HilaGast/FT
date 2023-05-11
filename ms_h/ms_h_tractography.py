from Tractography.connectivity_matrices import ConMat, WeightConMat
from Tractography.fiber_tracking import fiber_tracking_parameters, Tractography
import glob, os
from Tractography.files_loading import load_ft

main_fol = 'F:\Hila\TDI\siemens'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')

experiments = ['D31d18','D45d13','D60d11']
experiment = experiments[0]
tissue_labels_file_name = 'mprage_reg_seg.nii'
atlases = ['bnacor', 'yeo7_200']
for fol in all_subj_fol:
    if 'group' in fol or 'surfaces' in fol:
        continue
    exp_fol = f'{fol}{experiment}{os.sep}'
    dat = f'{exp_fol}diff_corrected_{experiment}.nii'
    streamlines = None
    if not os.path.exists(exp_fol + f'streamlines{os.sep}wb_csd_fa.tck'):
        parameters_dict = fiber_tracking_parameters(max_angle=30, sh_order=6, seed_density=4,
                                                    streamlines_lengths_mm=[50, 500], step_size=1, fa_th=.18)
        tracts = Tractography(exp_fol, 'csd', 'fa', 'wb', parameters_dict, dat,
                              tissue_labels_file_name=tissue_labels_file_name)
        tracts.fiber_tracking()
        streamlines = tracts.streamlines

    for atlas in atlases:
        if not os.path.exists(f'{exp_fol}cm{os.sep}num_{atlas}_cm_ord.npy'):
            if not streamlines:
                tract_name = os.path.join(fol,experiment,'streamlines','wb_csd_fa.tck')
                streamlines = load_ft(tract_name, dat)
            cm = ConMat(atlas=atlas, diff_file = dat, subj_folder = exp_fol, tract_name='wb_csd_fa.tck', streamlines=streamlines)
            cm.save_cm(fig_name=f'num_{atlas}', mat_type='cm_ord')


        if not os.path.exists(f'{exp_fol}cm{os.sep}add_{atlas}_cm_ord.npy'):
            if not streamlines:
                tract_name = os.path.join(fol, experiment, 'streamlines', 'wb_csd_fa.tck')
                streamlines = load_ft(tract_name, dat)
            cm = WeightConMat(weight_by='ADD', atlas=atlas, diff_file=dat, subj_folder=exp_fol,
                              tract_name='wb_csd_fa.tck', streamlines=streamlines)
            cm.save_cm(fig_name=f'add_{atlas}', mat_type='cm_ord')

        if not os.path.exists(f'{exp_fol}cm{os.sep}dist_{atlas}_cm_ord.npy'):
            if not streamlines:
                tract_name = os.path.join(fol, experiment, 'streamlines', 'wb_csd_fa.tck')
                streamlines = load_ft(tract_name, dat)
            cm = WeightConMat(weight_by='dist', atlas=atlas, diff_file=dat, subj_folder=exp_fol,
                              tract_name='wb_csd_fa.tck', streamlines=streamlines)
            cm.save_cm(fig_name=f'dist_{atlas}', mat_type='cm_ord')

