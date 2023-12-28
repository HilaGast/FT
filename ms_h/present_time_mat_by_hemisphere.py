import numpy as np
import matplotlib.pyplot as plt
def divide_mat_to_inter_intra_hemi_mats(mat, atlas):

    mat_intra = np.zeros(mat.shape)
    mat_inter = np.zeros(mat.shape)
    if atlas == 'bnacor':
        l_idx = np.arange(0, 105)
        r_idx = np.arange(105, 210)

    if atlas == 'yeo7_200':
        l_idx = np.arange(0, 100)
        r_idx = np.arange(100, 200)

    if atlas == 'yeo7_100':
        l_idx = np.arange(0, 50)
        r_idx = np.arange(50, 100)

    for c in range(mat.shape[0]):
        for r in range(mat.shape[1]):
            if c in l_idx and r in l_idx:
                mat_intra[c, r] = mat[c, r]
            elif c in r_idx and r in r_idx:
                mat_intra[c, r] = mat[c, r]
            else:
                mat_inter[c, r] = mat[c, r]

    return mat_intra, mat_inter

def show_inter_intra_hemi_hist(mat_intra, mat_inter, mat =None):
    mat_intra[mat_intra == 0] = np.nan
    mat_inter[mat_inter == 0] = np.nan
    mat[mat==0] = np.nan

    plt.hist(mat_intra[~np.isnan(mat_intra)], bins=50, color='blue', alpha = 0.2, range=(0, 100))
    plt.hist(mat_inter[~np.isnan(mat_inter)], bins=50, color='red', alpha = 0.2, range=(0, 100))
    plt.hist(mat_intra[~np.isnan(mat_intra)], bins=50, histtype='step', color='blue', linewidth=2, range=(0, 100))
    plt.hist(mat_inter[~np.isnan(mat_inter)], bins=50, histtype='step', color='red', linewidth=2, range=(0, 100))
    plt.legend(['Intra-hemispheric links', 'Inter-hemispheric links'])
    plt.show()


    plt.hist(mat[~np.isnan(mat)], bins=50, color='green', alpha = 0.2, range=(0, 100))
    plt.hist(mat[~np.isnan(mat)], bins=50, histtype='step', color='green', linewidth=2, range=(0, 100))
    plt.show()


if __name__ == '__main__':
    # mat = np.load(r'G:\data\V7\HCP\cm\median_TDI_EucSym_yeo7_100_D31d18.npy')
    mat = np.load(r'F:\Hila\TDI\siemens\D31d18\group_cm\mean_TDI_EucSym_yeo7_100_D31d18_h.npy')
    intra, inter = divide_mat_to_inter_intra_hemi_mats(mat, 'yeo7_100')

    show_inter_intra_hemi_hist(intra, inter, mat)
