from HCP_network_analysis.HCP_cm import euclidean_distance_matrix
from HCP_network_analysis.HCP_cm.euclidean_distance_matrix import find_labels_file, find_labels_centroids, euc_dist_mat
from Tractography.connectivity_matrices import ConMat, WeightConMat
from Tractography.fiber_tracking import fiber_tracking_parameters, Tractography
import glob, os
from Tractography.files_loading import load_ft
import numpy as np

main_fol = 'F:\Hila\TDI\siemens'
all_subj_fol = glob.glob(f'{main_fol}{os.sep}[C,T]*{os.sep}')

experiments = ['D31d18','D45d13','D60d11']
tissue_labels_file_name = 'rMPRAGE_brain_seg.nii'
atlases = ['yeo7_100','bnacor', 'yeo7_200']
for experiment in experiments:
    for fol in all_subj_fol:
        if 'tables' in fol or 'surfaces' in fol:
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

        if not os.path.exists(exp_fol + f'streamlines{os.sep}wb_csd_act.tck'):
            parameters_dict = fiber_tracking_parameters(max_angle=30, sh_order=8, seed_density=4,
                                                        streamlines_lengths_mm=[30, 500], step_size=0.5, fa_th=.18)
            tracts = Tractography(exp_fol, 'csd', 'act', 'wb', parameters_dict, dat,
                                  tissue_labels_file_name=tissue_labels_file_name,)
            tracts.fiber_tracking()
            streamlines = tracts.streamlines

        for atlas in atlases:
            if not os.path.exists(f'{exp_fol}cm{os.sep}num_{atlas}_cm_ord.npy'):
                if not streamlines:
                    tract_name = os.path.join(fol, experiment, 'streamlines', 'wb_csd_fa.tck')
                    streamlines = load_ft(tract_name, dat)
                cm = ConMat(atlas=atlas, diff_file=dat, subj_folder=exp_fol, tract_name='wb_csd_fa.tck',
                            streamlines=streamlines)
                cm.save_cm(fig_name=f'num_{atlas}', mat_type='cm_ord')

            if not os.path.exists(f'{exp_fol}cm{os.sep}add_{atlas}_cm_ord.npy'):
                if not streamlines:
                    tract_name = os.path.join(fol, experiment, 'streamlines', 'wb_csd_fa.tck')
                    streamlines = load_ft(tract_name, dat)
                cm = WeightConMat(weight_by='ADD', atlas=atlas, diff_file=dat, subj_folder=exp_fol,
                                  tract_name='wb_csd_fa.tck', streamlines=streamlines)
                cm.save_cm(fig_name=f'add_{atlas}', mat_type='cm_ord')

            cm_name = f'{exp_fol}cm{os.sep}num_{atlas}_cm_ord.npy'
            euc_dist_cm_name =f'{exp_fol}cm{os.sep}EucDist_{atlas}_cm_ord.npy'
            if not os.path.exists(euc_dist_cm_name):
                cm = np.load(cm_name)
                labels_file_path = find_labels_file(cm_name)
                label_ctd = find_labels_centroids(labels_file_path)
                euc_mat = euc_dist_mat(label_ctd, cm)
                np.save(euc_dist_cm_name, euc_mat)

