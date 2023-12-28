from Tractography.connectivity_matrices import *
import glob,os

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
atlases = ['yeo7_200']
for atlas in atlases:
    for sl in subj_list:
        if not os.path.exists(f'{sl}cm{os.sep}{atlas}_Num_Org_SC_100k_cm_ord.npy') and os.path.exists(f'{sl}streamlines{os.sep}HCP_tracts_100k.tck'):
            print(sl)
            diff_file = 'data.nii.gz'
            try:
                cm = ConMat(atlas=atlas, diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts_100k.tck')
                cm.save_cm(fig_name=f'{atlas}_Num_Org_SC_100k', mat_type='cm_ord')
                # cm.draw_con_mat(mat_type='cm_ord', show=False)
            except FileNotFoundError:
                try:
                    diff_file = 'data.nii'
                    cm = ConMat(atlas=atlas, diff_file=diff_file, subj_folder=sl, tract_name='HCP_tracts_100k.tck')
                    cm.save_cm(fig_name=f'{atlas}_Num_Org_SC_100k', mat_type='cm_ord')
                    # cm.draw_con_mat(mat_type='cm_ord', show=False)
                except FileNotFoundError:
                    continue
        if not os.path.exists(f'{sl}cm{os.sep}{atlas}_FA_Org_SC_100k_cm_ord.npy') and os.path.exists(f'{sl}streamlines{os.sep}HCP_tracts_100k.tck'):
            cm = WeightConMat(weight_by='data_FA', atlas=atlas, diff_file=diff_file, subj_folder=sl,
                              tract_name='HCP_tracts_100k.tck')
            cm.save_cm(fig_name=f'{atlas}_FA_Org_SC_100k', mat_type='cm_ord')
            # cm.draw_con_mat(mat_type='cm_ord',show=False)

        if not os.path.exists(f'{sl}cm{os.sep}{atlas}_ADD_Org_SC_100k_cm_ord.npy') and os.path.exists(f'{sl}streamlines{os.sep}HCP_tracts_100k.tck'):
            cm = WeightConMat(weight_by='data_3_2_AxPasi7', atlas=atlas, diff_file=diff_file, subj_folder=sl,
                              tract_name='HCP_tracts_100k.tck')
            cm.save_cm(fig_name=f'{atlas}_ADD_Org_SC_100k', mat_type='cm_ord')
            # cm.draw_con_mat(mat_type='cm_ord',show=False)

        # if not os.path.exists(f'{sl}cm{os.sep}{atlas}_Dist_Org_SC_cm_ord.npy'):
        #     diff_file = 'data.nii.gz'
        #     try:
        #         cm = WeightConMat(weight_by='dist', atlas=atlas, diff_file=diff_file, subj_folder=sl,
        #                       tract_name='HCP_tracts.tck')
        #     except FileNotFoundError:
        #         diff_file = 'data.nii'
        #         cm = WeightConMat(weight_by='dist', atlas=atlas, diff_file=diff_file, subj_folder=sl,
        #                           tract_name='HCP_tracts.tck')
        #
        #     cm.save_cm(fig_name=f'{atlas}_Dist_Org_SC', mat_type='cm_ord')



