from balance.ADD_before_vs_after import *
from calc_corr_statistics.pearson_r_calc import *

def load_rms_subj(subj_folder_list, table_file_name = r'F:\Hila\balance\Balance Study subjects.xlsx', sheet='All'):
    import pandas as pd

    t1 = pd.read_excel(table_file_name,sheet)
    rms=[]

    for s in subj_folder_list:
        sn = str.split(s,os.sep)[-1]
        state = str.split(s,os.sep)[-3]
        snum = int(sn[-2::])

        rms.append(float(t1['balance improvement (ΔRMS)'][(t1['index '] == snum) & (t1['state'] == state)].values))

    return rms



if __name__ == '__main__':
    wt ='ADD'
    main_folder = r'F:\data\balance'
    atlas_name = 'bna'
    atlas_main_folder = r'F:\data\atlases\BNA'

    before_subj = glob.glob(main_folder + f'{os.sep}e*{os.sep}before{os.sep}*')
    after_subj = glob.glob(main_folder + f'{os.sep}e*{os.sep}after{os.sep}*')

    rms = load_rms_subj(before_subj)

    before_vol_mat, mni_atlas_file_name, idx, subj_idx =  volume_based_var(atlas_name, wt, atlas_main_folder, before_subj)
    after_vol_mat, mni_atlas_file_name, idx, subj_idx =  volume_based_var(atlas_name, wt, atlas_main_folder, after_subj)

    delta_vol_mat = np.absolute(after_vol_mat-before_vol_mat)

    r, p = calc_corr(rms, delta_vol_mat, fdr_correct=False, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'deltaRMS_vs_deltaADD_r_th_'+atlas_name, main_folder)

    r, p = calc_corr(rms, delta_vol_mat, fdr_correct=True, remove_outliers=True)
    weighted_by_atlas,weights_dict = weight_atlas_by_add(mni_atlas_file_name,r,idx)
    save_as_nii(weighted_by_atlas, mni_atlas_file_name, f'deltaRMS_vs_deltaADD_r_th_fdr_'+atlas_name, main_folder)




