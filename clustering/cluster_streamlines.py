from dipy.tracking.streamline import Streamlines
from dipy.viz import window, actor
from FT.weighted_tracts import *
from dipy.segment.clustering import QuickBundles
from FT.single_fascicle_vizualization import choose_specific_bundle
from FT.remove_cci_outliers import remove_cci_outliers

main_folder = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep'
folder_name = main_folder + all_subj_folders[0]
n = all_subj_names[0]
nii_file = load_dwi_files(folder_name)[5]

tract_path = folder_name + r'\streamlines' + n + '_wholebrain_3d_plus_new.trk'

streamlines = load_ft(tract_path, nii_file)
lab_labels_index, affine = nodes_by_index_mega(folder_name)
masked_streamlines = choose_specific_bundle(streamlines, affine, folder_name, mask_type='slf')
streamlines = Streamlines(masked_streamlines)
s,s_i = remove_cci_outliers(streamlines)
qb = QuickBundles(threshold=25.)
clusters = qb.cluster(s)

print("Nb. clusters:", len(clusters))
print("Cluster sizes:", map(len, clusters))
print("Small clusters:", clusters < 10)
print("Streamlines indices of the first cluster:\n", clusters[0].indices)
print("Centroid of the last cluster:\n", clusters[-1].centroid)

interactive = True

ren = window.Renderer()
'''
ren.SetBackground(1, 1, 1)
ren.add(actor.streamtube(streamlines, window.colors.white))
if interactive:
    window.show(ren)
'''

colormap = actor.create_colormap(np.arange(len(clusters)))

window.clear(ren)
ren.SetBackground(1, 1, 1)
ren.add(actor.streamtube(s, window.colors.white, opacity=0.05))
ren.add(actor.streamtube([clusters.centroids[i] for i in np.where(clusters>9)[0]], colormap, linewidth=0.4))
if interactive:
    window.show(ren)