from ms_h.extract_values_2_table import load_cc_mask
import glob, os
import pandas as pd
import nibabel as nib
import numpy as np
from pingouin import mixed_anova, pairwise_ttests


def find_group_by_name(subj_name):
    if subj_name.startswith('C'):
        group = 'H'
    elif subj_name.startswith('T'):
        group = 'MS'
    else:
        group = 'UnKnown'

    return group



if __name__ == '__main__':
    main_fol = 'Y:\qnap\siemens'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}AxCaliber{os.sep}')
    all_subj_names = [s.split(os.sep)[3] for s in all_subj_fol]
    vol_name = ['CC vol','FA', 'MD', 'ADD', 'pFR','pH','pCSF'][3]
    parts = ["G","AB","MB","PB","Sp"]
    experiments = ['D60d11', 'D45d13', 'D31d18']

    for experiment in experiments:
        main_table = pd.DataFrame(columns=['Value','Group','Part','Subject'])
        scan_name = f"diff_corrected_{experiment}"

        for subj_fol, subj_name in zip(all_subj_fol, all_subj_names):
            group = find_group_by_name(subj_name)
            cc_mask, slice_num = load_cc_mask(subj_fol, scan_name)
            if vol_name == 'CC vol':
                from ms_h.extract_values_2_table import compute_mask_sizes

                vec_vol = compute_mask_sizes(cc_mask)
                vec_vol = list(map(float, vec_vol))
            else:
                from ms_h.extract_values_2_table import load_vol, compute_vol_from_mask

                vol_mat = load_vol(subj_fol, vol_name, scan_name)
                vec_vol = compute_vol_from_mask(vol_mat, cc_mask, slice_num)

            subj_table = pd.DataFrame(
                {'Value': vec_vol[1:], 'Group': group, 'Part': parts, 'Subject': len(parts) * subj_name})
            main_table = main_table.append(subj_table, ignore_index=True)

        anova_results = main_table.mixed_anova(dv='Value', between='Group', within='Part', subject='Subject')
        pairwise_results = main_table.pairwise_ttests(dv='Value', between='Group', within='Part', subject='Subject')
        anova_results.to_excel(
            f'Y:\qnap\siemens_results_CC{os.sep}mixedANOVA_{vol_name}_X_Parts_{scan_name.split("_")[2]}.xlsx')
        pairwise_results.to_excel(
            f'Y:\qnap\siemens_results_CC{os.sep}pairwise_{vol_name}_X_Parts_{scan_name.split("_")[2]}.xlsx')
