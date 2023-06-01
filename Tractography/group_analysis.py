import numpy as np

def create_all_subject_connectivity_matrices(subjects):

    connectivity_matrices = []
    for subject in subjects:
        connectivity_matrices.append(np.load(subject))
    connectivity_matrices = np.array(connectivity_matrices)
    connectivity_matrices = np.swapaxes(connectivity_matrices, 0, -1)

    return connectivity_matrices

def norm_matrices(matrices, norm_type = 'scaling'):
    norm_matrices = matrices.copy()
    for s in range(matrices.shape[-1]):
        if norm_type == 'scaling':
            norm_matrices[norm_matrices==0] = np.nan
            norm_matrices[:,:,s] = norm_scaling(matrices[:,:,s])
        elif norm_type == 'fisher':
            norm_matrices[norm_matrices == 0] = np.nan
            norm_matrices[:,:,s] = fisher_transformation(matrices[:,:,s])
        elif norm_type == 'z-score':
            norm_matrices[norm_matrices == 0] = np.nan
            norm_matrices[:,:,s] = z_score(matrices[:,:,s])
        elif norm_type == 'rating':
            norm_matrices[:,:,s] = rating(matrices[:,:,s])

    return norm_matrices

def norm_scaling(matrix):
    norm_mat = (matrix - np.nanmin(matrix)) / (np.nanmax(matrix) - np.nanmin(matrix))
    return norm_mat

def fisher_transformation(matrix):
    matrix = np.arctanh(matrix)
    return matrix

def z_score(matrix):
    matrix = (matrix - np.nanmean(matrix)) / np.nanstd(matrix)
    return matrix

def rating(matrix):
    from scipy.stats import rankdata
    matrix = rankdata(matrix, method='dense').reshape(matrix.shape)-1
    return matrix