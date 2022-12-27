import matplotlib.pyplot as plt
import nibabel as nib
import scipy.ndimage as ndimage
import numpy as np

brain = nib.load(r'F:\Hila\Ax3D_Pack\V6\v7calibration\TheBase4Ever\YA_lab_Yaniv_002417_20210309_1521\r20210309_152115T1wMPRAGERLs008a1001_brain.nii').get_fdata()
overlay = nib.load(r'F:\Hila\Ax3D_Pack\V6\v7calibration\TheBase4Ever\YA_lab_Yaniv_002417_20210309_1521\ADD_along_streamlines_WMmasked.nii').get_fdata()
i=62 #or slice 62

rot_brain = np.swapaxes(brain[i,:,:],1,0)
rot_overlay = np.swapaxes(overlay[i,:,:],1,0)
rot_brain = rot_brain[::-1,::-1]
rot_overlay = rot_overlay[::-1,::-1]

rot_overlay[rot_overlay<5] = np.nan

rot_overlay[0:30,:] = np.nan
rot_overlay[47:,:] = np.nan
rot_overlay[:,82:] = np.nan
rot_overlay[41:49,50:73] = np.nan

fig, ax = plt.subplots()

im1 = ax.imshow(rot_brain,cmap=plt.cm.get_cmap('gray'))

im2 = ax.imshow(rot_overlay,cmap=plt.cm.get_cmap('hot').reversed(),vmin=6,vmax=12) #or 8 to 10
cbar = ax.figure.colorbar(im2, ax=ax, ticks=[],shrink=0.7, aspect=15)
plt.tight_layout()
plt.axis('off')
plt.show()