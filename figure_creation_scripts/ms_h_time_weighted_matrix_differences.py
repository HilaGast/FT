import numpy as np
import matplotlib.pyplot as plt

h_mat = np.load('F:\Hila\siemens\median_time_th30_bnacor_D31d18_h.npy')
ms_mat = np.load('F:\Hila\siemens\median_time_th30_bnacor_D31d18_ms.npy')

diff_mat = h_mat - ms_mat

plt.imshow(diff_mat,cmap='seismic',vmin=-0.05, vmax = 0.05)

plt.colorbar()

plt.show()