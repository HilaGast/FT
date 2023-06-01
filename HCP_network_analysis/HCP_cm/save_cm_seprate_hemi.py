import numpy as np

atlas = 'yeo7_100'
mat_type= 'Dist'

matrix = np.load(f'G:\data\V7\HCP\cm\median_{atlas}_{mat_type}_Org_SC.npy')
num_of_nodes = matrix.shape[0]
matrix_lh = matrix[:num_of_nodes//2,:num_of_nodes//2]
matrix_rh = matrix[num_of_nodes//2:,num_of_nodes//2:]

np.save(f'G:\data\V7\HCP\cm\median_{atlas}_{mat_type}_Org_SC_LH.npy', matrix_lh)
np.save(f'G:\data\V7\HCP\cm\median_{atlas}_{mat_type}_Org_SC_RH.npy', matrix_rh)