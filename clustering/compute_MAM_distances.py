from dipy.segment.bundles import bundles_distances_mam
m=bundles_distances_mam(streamlines,streamlines,'avg')

sum_of_m = np.sum(m,axis=0)

sum_of_m = list(sum_of_m)

imax = sum_of_m.index(max(sum_of_m))
imin = sum_of_m.index(min(sum_of_m))
farest =m[:,imax]+m[:,imin]
farest = list(farest)
ifar = farest.index(max(farest))
iclose1 = farest.index(min(farest))
cl2 = list(m[:,ifar]+m[:,imin])
iclose2 = cl2.index(min(cl2))

#locs = np.asarray([m[:,imin],m[:,imax],m[:,ifar],m[:,iclose1],m[:,iclose2]]).T
locs = np.asarray([m[:,imin],m[:,imax],m[:,ifar]]).T
