import matplotlib.pyplot as plt
import numpy as np
g = np.load(r'C:\Users\Admin\my_scripts\Ax3D_Pack\mean_vals\mean_genu_AxCaliber_nonnorm.npy')
gm = np.nanmean(g[g>0])
gs = np.nanstd(g[g>0])
b = np.load(r'C:\Users\Admin\my_scripts\Ax3D_Pack\mean_vals\mean_body_AxCaliber_nonnorm.npy')
bm = np.nanmean(b[b>0])
bs = np.nanstd(b[b>0])
s = np.load(r'C:\Users\Admin\my_scripts\Ax3D_Pack\mean_vals\mean_splenium_AxCaliber_nonnorm.npy')
sm = np.nanmean(s[s>0])
ss = np.nanstd(s[s>0])
plt.bar([0,1,2],[gm,bm,sm])
plt.show()
