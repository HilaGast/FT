import glob,os
import numpy as np


def calc_group_average_mat(all_files_name, atlas, type='mean'):
    mutual_mat = []
    for cm_name in all_files_name:
        cm = np.load(cm_name)
        mutual_mat.append(cm)
    idx = np.load(f'{os.path.dirname(cm_name)}{os.sep}{atlas}_cm_ord_lookup.npy')
    mutual_mat = np.asarray(mutual_mat)
    mask_mat = mutual_mat > 0
    mask_mat = np.sum(mask_mat, axis=0) / mask_mat.shape[0]
    mask_mat = mask_mat>=0.2 #choose only edges that appeared in >x% of subjects usually 0.2
    mean_mat=np.zeros(mask_mat.shape)

    if type == 'mean':
        for row in range(mean_mat.shape[0]):
            for col in range(row + 1):
                mat_vec = mutual_mat[:, row, col]
                mean_mat[row, col] = np.nanmean(mat_vec[mat_vec > 0])
                mean_mat[col, row] = np.nanmean(mat_vec[mat_vec > 0])
    elif type == 'median':
        for row in range(mean_mat.shape[0]):
            for col in range(row + 1):
                mat_vec = mutual_mat[:, row, col]
                mean_mat[row, col] = np.nanmedian(mat_vec[mat_vec > 0])
                mean_mat[col, row] = np.nanmedian(mat_vec[mat_vec > 0])

    return mean_mat


if __name__ == '__main__':

    subj_list = glob.glob(rf'G:\data\V7\HCP\*[0-9]{os.sep}')
    # weights = ['Num_Org', 'Num_HistMatch', 'FA_Org', 'FA_HistMatch', 'ADD_Org', 'ADD_HistMatch', 'Dist_Org',
    #            'Dist_HistMatch']
    weights = ['ADD', 'FA', 'Num']
    th = 'HistMatch'
    atlases = ['yeo7_200']#, 'bna']
    ncm_options = ['SC']#, 'SPE']

    for atlas in atlases:
        for ncm in ncm_options:
            for w in weights:
                file_names = glob.glob(rf'G:\data\V7\HCP\*[0-9]{os.sep}cm{os.sep}{atlas}_{w}_{th}_{ncm}_cm_ord.npy')
                average_mat = calc_group_average_mat(file_names, atlas)

                # np.save(rf'G:\data\V7\HCP\cm\{atlas}_cm_ord_lookup', idx)
                np.save(rf'G:\data\V7\HCP\cm\average_{atlas}_{w}_{th}_{ncm}_atleast_half_subjects.npy', average_mat)




