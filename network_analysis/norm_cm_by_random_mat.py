import numpy as np
import random


def rand_mat(cm, type):

    if type=='links shuffle' or type=='links_shuffle':
        cm_rand = links_shuffle(cm)

    elif type=='weights shuffle' or type=='weights_shuffle':
        cm_rand = weights_shuffle(cm)

    return cm_rand


def links_shuffle(cm):
    cm1 = np.copy(cm)
    shape = np.shape(cm1)
    tril = np.ravel_multi_index(np.tril_indices(len(cm1), k=0), np.shape(cm1)) #k = diagonal offset
    indices = np.copy(tril)
    random.shuffle(indices)
    cm_rand = np.zeros(np.shape(cm1))
    cm_rand = cm_rand.reshape(-1)
    cm1 = cm1.reshape(-1)
    cm_rand[tril] = cm1[indices]
    cm_rand = cm_rand.reshape(shape)
    cm_rand = np.tril(cm_rand) + np.triu(cm_rand.T, 1)

    return cm_rand


def weights_shuffle(cm):
    cm1 = np.copy(cm)
    shape = np.shape(cm1)
    tril = np.ravel_multi_index(np.tril_indices(len(cm1), k=0), np.shape(cm1)) #k = diagonal offset
    cm1 = cm1.reshape(-1)
    tril_vals = cm1[tril]
    tril_edge_locs = tril_vals>0
    indices = np.copy(tril[ tril_edge_locs]) #choose only existing edges
    random.shuffle(indices)
    cm_rand = np.zeros(np.shape(cm1))
    cm_rand[tril[ tril_edge_locs]] = cm1[indices]
    cm_rand = cm_rand.reshape(shape)
    cm_rand = np.tril(cm_rand) + np.triu(cm_rand.T, 1)

    return cm_rand


def make_n_rand_mat(cm,n,type):
    rand_cm = np.zeros((cm.shape[0],cm.shape[1],n))
    for i in range(0,n):
        rand_cm_i = rand_mat(cm, type)
        rand_cm[:,:,i] = rand_cm_i

    return rand_cm