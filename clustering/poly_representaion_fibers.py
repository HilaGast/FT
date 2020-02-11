import numpy as np
from scipy.linalg import pinv

def distance_vec_rep_of_fibers(fi):
    '''This function calculates the distance of each point on the fiber fr m th first point
        Input:
        fi - a (n,3) np.ndarray of a single fiber. n is the number of points that represent the fiber
        Output:
        dist_vec - a (,n) column vec of distance represntation of the fiber'''

    p1 = fi[0,:]
    dist_vec = np.zeros(fi.shape[0])
    for pi,i in zip(fi,range(fi.shape[0])):
        disti = np.linalg.norm(p1-pi)
        dist_vec[i] = disti

    return dist_vec


def distance_powered_matrix(dist_vec, degree):
    '''This function calculates the matrix to interpolate polynomial function for X,Y & Z of each fiber.
    it takes the distance representation vector and power it according to the chosen degree.
    Input:
    dist_vec - a (,n) column vec of distance represntation of the fiber
    degree - the polynomial degree wanted
    Output:
    dist_mat - a (n, degree+1) np.ndarray of fiber points an their calculated powers'''
    dist_mat = np.zeros([len(dist_vec), degree+1])
    for i in range(degree+1):
        dist_mat[:,i] = dist_vec.T**i

    return dist_mat


def least_squares_poly_rep(fi,comp,dist_mat):
    '''This function calculates the least square polynomial function for a single component of the fiber
    Calculates the follow Eq: poly_vec = (dist_mat.T * dist_mat).pinv * dist_mat.T * fi[:,comp]
    Input:
    fi - a (n,3) np.ndarray of a single fiber. n is the number of points that represent the fiber
    comp - {'X','Y','Z'} is the current component for polynomial calculation
    dist_mat - a (n, degree+1) np.ndarray of fiber points an their calculated powers
    Output:
    poly_vec - a (,degree+1) vec representation of the polynomial parameters
    '''
    if comp == 'X':
        ax = 0
    elif comp == 'Y':
        ax = 1
    elif comp == 'Z':
        ax = 2

    dup_mat = np.matmul(dist_mat.T, dist_mat)
    inv_dup_mat = pinv(dup_mat)
    poly_vec = np.matmul(np.matmul(inv_dup_mat, dist_mat.T), fi[:,ax])

    return poly_vec


def poly_xyz_vec_calc(fi, degree=3):
    ''''''
    dist_vec = distance_vec_rep_of_fibers(fi)
    dist_mat = distance_powered_matrix(dist_vec,degree)
    poly_vec_x = least_squares_poly_rep(fi,'X',dist_mat)
    poly_vec_y = least_squares_poly_rep(fi,'Y',dist_mat)
    poly_vec_z = least_squares_poly_rep(fi,'Z',dist_mat)

    poly_xyz = np.concatenate([poly_vec_x,poly_vec_y,poly_vec_z],0)

    return poly_xyz
