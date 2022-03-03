import glob
from Tractography.fiber_tracking import *
from Tractography.tractography_vis import show_tracts_by_mask


main_folder = r'F:\Hila\tractography_comparisons'
subj_folders = glob.glob(f'{main_folder}{os.sep}sub*{os.sep}')

tissue_labels_file_name = "dseg.nii.gz"
pve_file_name = "probseg.nii.gz"

for sf in subj_folders:
    ses_fold = sorted(glob.glob(fr'{sf}{os.sep}ses*'))[0]
    subj_main_folder = f'{ses_fold}{os.sep}dwi{os.sep}'
    dat = glob.glob(f'{ses_fold}{os.sep}dwi{os.sep}*_dir-FWD_space-orig_desc-preproc_dwi.nii.gz')[0]

    # CSD Tractography:
    if not os.path.exists(subj_main_folder+f'streamlines{os.sep}wb_csd_act.tck'):
        parameters_dict = fiber_tracking_parameters(max_angle=30 ,sh_order= 8, seed_density= 4, streamlines_lengths_mm= [30, 500], step_size= 0.5, fa_th= .18)
        trk2 = Tractography(subj_main_folder, 'csd' , 'act' , 'wb' , parameters_dict, dat, tissue_labels_file_name=tissue_labels_file_name, pve_file_name=pve_file_name)
        trk2.fiber_tracking()

    if not os.path.exists(subj_main_folder + f'streamlines{os.sep}wb_csd_fa.tck'):
        parameters_dict = fiber_tracking_parameters(max_angle=30, sh_order=8, seed_density=4,
                                                    streamlines_lengths_mm=[30, 500], step_size=0.5, fa_th=.18)
        trk2 = Tractography(subj_main_folder, 'csd', 'fa', 'wb', parameters_dict, dat,
                            tissue_labels_file_name=tissue_labels_file_name, pve_file_name=pve_file_name)
        trk2.fiber_tracking()

    # MSMT Tractography:
    if not os.path.exists(subj_main_folder+f'streamlines{os.sep}wb_msmt_cmc.tck'):
        parameters_dict = fiber_tracking_parameters(max_angle= 30,sh_order= 8, seed_density= 3, streamlines_lengths_mm= [50, 1000], step_size= 0.2)
        trk1 = Tractography(subj_main_folder, 'msmt' , 'cmc' , 'wm' , parameters_dict, dat, tissue_labels_file_name=tissue_labels_file_name, pve_file_name=pve_file_name)
        trk1.fiber_tracking()

'''


for sf in subj_folders[3::]:
    ses_fold = sorted(glob.glob(fr'{sf}{os.sep}ses*'))[0]
    subj_main_folder = f'{ses_fold}{os.sep}dwi{os.sep}'
    diff_file = glob.glob(f'{ses_fold}{os.sep}dwi{os.sep}*_dir-FWD_space-orig_desc-preproc_dwi.nii.gz')[0]
    nii_ref = os.path.join(subj_main_folder, diff_file)
    affine = nib.load(nii_ref).affine

    csd_tract_file = subj_main_folder + f'streamlines{os.sep}wb_csd_act.tck'
    streamlines = load_ft(csd_tract_file, nii_ref)
    s_list = [s1 for s1 in streamlines]
    show_tracts_by_mask(subj_main_folder, 'cc_mask', s_list, affine, fig_type='cc_csd')

    msmt_tract_file = subj_main_folder+f'streamlines{os.sep}wb_msmt_cmc.tck'
    streamlines = load_ft(msmt_tract_file, nii_ref)
    s_list = [s1 for s1 in streamlines]
    show_tracts_by_mask(subj_main_folder, 'cc_mask', s_list, affine, fig_type='cc_msmt', downsamp=5)

'''
