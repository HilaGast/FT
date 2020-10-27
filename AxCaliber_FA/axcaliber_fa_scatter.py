import numpy as np
import matplotlib.pyplot as plt

ax = np.load(r'F:\Hila\Ax3D_Pack\mean_vals\aal3_atlas\mean_weighted_mega_wholebrain_4d_labmask_aal3_nonnorm.npy')
fa = np.load(r'F:\Hila\Ax3D_Pack\mean_vals\aal3_atlas\mean_weighted_mega_wholebrain_4d_labmask_aal3_FA_nonnorm.npy')

#intra-hemi:
falr=[]
axlr=[]
facom=[]
axcom=[]

for i in range(78):
    for j in range(i+1):
        falr.append(fa[i,j])
        axlr.append(ax[i,j])

for i in range(88,166):
    for j in range(88,i+1):
        falr.append(fa[i,j])
        axlr.append(ax[i,j])

for i in range(78):
    for j in range(88,166):
        facom.append(fa[i,j])
        axcom.append(ax[i,j])

falr = np.asarray(falr)
axlr = np.asarray(axlr)
facom = np.asarray(facom)
axcom = np.asarray(axcom)

falr1 = falr[axlr != 0]
falr1 = falr1/100
axlr1 = axlr[axlr !=0]
facom1 = facom[axcom !=0]
facom1 = facom1/100
axcom1 = axcom[axcom != 0]

ax1 = plt.subplot()
ax1.set_title('FA - AxCaliber')
ax1.set_xlabel('AxCaliber')
ax1.set_ylabel('FA')
ax1.scatter(axcom1,facom1,1,c='blue')
ax1.scatter(axlr1,falr1,1,c='red')

plt.show()

