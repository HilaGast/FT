import matplotlib.pyplot as plt
import nibabel as nib
import scipy.ndimage as ndimage
import numpy as np

brain = nib.load(r'F:\data\V7\TheBase4Ever\YA_lab_Yaniv_002044_20201025_0845\r20201025_084555T1wMPRAGERLs004a1001_brain.nii').get_fdata()
overlay = nib.load(r'F:\data\V7\TheBase4Ever\YA_lab_Yaniv_002044_20201025_0845\002044_ADD_along_streamlines_WMmasked.nii').get_fdata()
i=63 #or slice 62
rot_brain = ndimage.rotate(brain[i,:,:],90,reshape=True)
rot_overlay = ndimage.rotate(overlay[i,:,:],90,reshape=True)
rot_overlay[rot_overlay<5] = np.nan


fig, ax = plt.subplots()

im1 = ax.imshow(rot_brain,cmap=plt.cm.get_cmap('gray'))

im2 = ax.imshow(rot_overlay,cmap=plt.cm.get_cmap('hot').reversed(),vmin=7,vmax=11) #or 8 to 10
cbar = ax.figure.colorbar(im2, ax=ax, ticks=[],shrink=0.7, aspect=15)
plt.tight_layout()
plt.axis('off')
plt.show()