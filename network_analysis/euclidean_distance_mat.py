import nibabel as nib
import numpy as np
from skimage.measure import regionprops
from scipy.ndimage import zoom
def create_euclidean_dist_mat(atlas):
    if atlas == 'yeo7_200':
        img_name = r'G:\data\atlases\yeo\yeo7_200\yeo7_200_atlas.nii'

    img = nib.load(img_name).get_fdata()
    img = np.asarray(img, int)

    props = regionprops(img)
    clocs = []
    labels = []
    for i in range(1,np.unique(img).shape[0]):
        clocs.append(props[i-1].centroid)
        labels.append(props[i-1].label)

    dist_mat = np.zeros([len(labels),len(labels)])
    for loc1, lab1 in zip(clocs, labels):
        for loc2, lab2 in zip(clocs, labels):
            dist = np.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2 + (loc1[2]-loc2[2])**2)
            dist_mat[lab1-1,lab2-1] = dist

    return dist_mat

