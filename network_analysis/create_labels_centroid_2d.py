import numpy as np
import nibabel as nib
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from networkx.drawing.layout import rescale_layout_dict

def create_nodes_position(atlas, slice='horizontal'):
    if atlas == 'aal3':
        img_file = r'F:\Hila\aal\aal3\AAL3_highres_atlas.nii'
    elif atlas == 'yeo7_200':
        img_file = r'F:\Hila\aal\yeo7_200\yeo7_200_atlas.nii'
    elif atlas == 'bna':
        img_file = r'G:\data\atlases\BNA\BN_Atlas_274_combined_1mm.nii'
    img_mat = nib.load(img_file)
    img = img_mat.get_fdata()
    img = np.asarray(img, int)
    lrgimg = zoom(img, 2, order=0, mode='nearest')
    props = regionprops(lrgimg)
    imgcent = np.zeros(lrgimg.shape)
    clocs = []
    labels = []
    for i in range(1,np.unique(lrgimg).shape[0]):
        clocs.append(props[i-1].centroid)
        labels.append(props[i-1].label)

    for loc, lab in zip(clocs, labels):
        imgcent[int(loc[0]), int(loc[1]), int(loc[2])] = lab

    imgcent = imgcent.astype(int)

    if slice == 'horizontal':
        #img2d = np.sum(imgcent, 2)  # horizontal
        img2d = horizontal_sum(imgcent,labels)
        #img2d = np.flipud(img2d)

    if atlas == 'aal3':
        img2d[img2d>81]-=2
        img2d[img2d>35]-=2
        labels = np.asarray(labels)
        labels[labels>81]-=2
        labels[labels > 35] -= 2
        labels = list(labels)

    positions = dict()
    for l in labels:
        idx = np.where(img2d == l)
        positions[l] = np.asarray([float(idx[0]), float(idx[1])])

    pos = rescale_layout_dict(positions, 1)

    return pos


def horizontal_sum(imgcent,labels):
    # Horizontal look:
    img2d = np.zeros((imgcent.shape[0], imgcent.shape[1]))

    xe = int(imgcent.shape[0] / 2)
    ye = int(imgcent.shape[1] / 2)
    for l in labels:
        idx = np.where(imgcent == l)
        xi = int(idx[0] + (idx[0] - xe) / 100)
        yi = int(idx[1] + (idx[1] - ye) / 100)
        if xi < 0:
            xi = 0
        elif xi > 2 * xe:
            xi = 2 * xe
        if yi < 0:
            yi = 0
        elif yi > 2 * ye:
            yi = 2 * ye

        while img2d[xi, yi]!=0:
            xi=xi+1
            yi=yi+1

        img2d[xi, yi] = l

    return img2d