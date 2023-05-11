import mantel
import numpy as np

import scipy.io as sio
vals = sio.loadmat(r'F:\Hila\work comp desktop backup\rat_scans\rat2\skeleton_cc_values.mat')['vals']
x1 = vals[0,:]
x2 = vals[1,:]

dist1 = np.zeros((len(x1),len(x1)))
for i in range(len(x1)):
    for j in range(len(x1)):
        dist1[i,j] = np.abs(x1[i]-x1[j])

dist2 = np.zeros((len(x2),len(x2)))
for i in range(len(x2)):
    for j in range(len(x2)):
        dist2[i,j] = np.abs(x2[i]-x2[j])

mantel_results = mantel.test(dist1, dist2)

x2 = vals[3,:]
dist2 = np.zeros((len(x2),len(x2)))
for i in range(len(x2)):
    for j in range(len(x2)):
        dist2[i,j] = np.abs(x2[i]-x2[j])

mantel_results = mantel.test(dist1, dist2)

x2 = vals[4,:]
dist2 = np.zeros((len(x2),len(x2)))
for i in range(len(x2)):
    for j in range(len(x2)):
        dist2[i,j] = np.abs(x2[i]-x2[j])

mantel_results = mantel.test(dist1, dist2)