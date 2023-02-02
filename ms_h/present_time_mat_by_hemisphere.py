import numpy as np
import matplotlib.pyplot as plt
def divide_mat_to_inter_intra_hemi_mats(mat, atlas):

    mat_intra = np.zeros(mat.shape)
    mat_inter = np.zeros(mat.shape)
    if atlas == 'bnacor':
        l_idx = np.arange(0, 105)
        r_idx = np.arange(105, 210)

        for c in range(mat.shape[0]):
            for r in range(mat.shape[1]):
                if c in l_idx and r in l_idx:
                    mat_intra[c, r] = mat[c, r]
                elif c in r_idx and r in r_idx:
                    mat_intra[c, r] = mat[c, r]
                else:
                    mat_inter[c, r] = mat[c, r]
    return mat_intra, mat_inter

def show_inter_intra_hemi_hist(mat_intra, mat_inter):
    mat_intra[mat_intra == 0] = np.nan
    mat_inter[mat_inter == 0] = np.nan

    plt.hist(mat_intra[~np.isnan(mat_intra)], bins=50, histtype='step', color='darkviolet', linewidth=2, range=(0, 0.15), density=True)
    plt.hist(mat_inter[~np.isnan(mat_inter)], bins=50, histtype='step', color='darkorange', linewidth=2, range=(0, 0.15), density=True)
    plt.show()


if __name__ == '__main__':
    mat = np.load(r'F:\Hila\siemens\median_time_th30_bnacor_D31d18_h.npy')
    mat_intra, mat_inter = divide_mat_to_inter_intra_hemi_mats(mat, 'bnacor')
    show_inter_intra_hemi_hist(mat_intra, mat_inter)