from HCP_network_analysis.cc_violin_HCP import *
from cc_analysis.cc_boxplot import *
from Tractography.fiber_weighting import weight_streamlines

def divide_2_groups(subj_list):
    idx_h = []
    idx_ms = []

    for i,sl in enumerate(subj_list):
        if sl.startswith('C'):
            idx_h.append(i)
        elif sl.startswith('T'):
            idx_ms.append(i)
        else:
            continue

    return idx_h, idx_ms

if __name__ == '__main__':
    main_fol = 'Y:\qnap\siemens'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}')
    all_subj_names = [s.split(os.sep)[3] for s in all_subj_fol]
    idx_h, idx_ms = divide_2_groups(all_subj_names)
    n_h = len(idx_h)
    n_ms = len(idx_ms)
    protocol_list = ['Healthy'] * n_h * 5 + ['MS'] * n_ms * 5
    parts_list = ['Genu','Anterior Body', 'Mid Body', 'Posterior Body', 'Splenium']*(n_h+n_ms)
    subji = [i for i in range(1,n_h+n_ms+1)]*5
    subji.sort()

    vol_names = ['FA', 'MD', 'ADD', 'pFr', 'pH', 'pCSF']
    v = vol_names[2]

    experiments = ['D60d11', 'D45d13', 'D31d18']
    e = experiments[1]

    val_vec = []
    for i in idx_h:
        sl = all_subj_fol[i]
        tract_file_name = f'{sl}{e}{os.sep}streamlines{os.sep}wb_csd_fa.tck'
        data_file_name = f'{sl}{e}{os.sep}diff_corrected_{e}.nii'
        affine = nib.load(data_file_name).affine
        streamlines = load_tck(tract_file_name, data_file_name)
        slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(f'{sl}AxCaliber{os.sep}diff_corrected_{e}_CC_mask.mat')


        for mask in [mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium]:
            val = calc_cc_part_val(streamlines, mask, affine, f'{sl}AxCaliber', calc_type='median', weight_by = f'{e}_{v}')
            val_vec.append(val)


    for i in idx_ms:
        sl = all_subj_fol[i]
        tract_file_name = f'{sl}{e}{os.sep}streamlines{os.sep}wb_csd_fa.tck'
        data_file_name = f'{sl}{e}{os.sep}diff_corrected_{e}.nii'
        affine = nib.load(data_file_name).affine
        streamlines = load_tck(tract_file_name, data_file_name)
        slice_num, mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium = load_mat_cc_file(f'{sl}diff_corrected_{e}_CC_mask.mat')


        for mask in [mask_genu, mask_abody, mask_mbody, mask_pbody, mask_splenium]:
            val = calc_cc_part_val(streamlines, mask, affine, f'{sl}AxCaliber', calc_type='median', weight_by = f'{e}_{v}')
            val_vec.append(val)

    d_vals = {'SubjNum':subji,'Protocol':protocol_list, 'CC_Part': parts_list,'ADD [\u03BCm]':val_vec}
    cc_parts_table = pd.DataFrame(d_vals)
    cc_parts_table, num_lo = detect_and_remove_outliers(cc_parts_table)
    print(cc_parts_table)
    print(f'Removed {num_lo} outliers')
    create_comperative_cc_vioplot(cc_parts_table, 'split')
    anova_for_different_protocols(cc_parts_table)


