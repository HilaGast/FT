import numpy as np
import matplotlib.pyplot as plt

h_mat = np.load('F:\Hila\TDI\siemens\group_cm\median_time_th3_bnacor_D60d11_h.npy')
ms_mat = np.load('F:\Hila\TDI\siemens\group_cm\median_time_th3_bnacor_D60d11_ms.npy')

diff_mat = h_mat - ms_mat

plt.imshow(diff_mat,cmap='seismic')

plt.colorbar()

plt.show()