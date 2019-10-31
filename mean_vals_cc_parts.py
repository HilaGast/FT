from FT.weighted_tracts import load_dwi_files, load_ft, weighting_streamlines, nodes_labels_mega
from FT.all_subj import all_subj_names, all_subj_folders
from dipy.viz import window, actor
from dipy.tracking.streamline import values_from_volume
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def cc_parts_viz(n, folder_name):
    tract_path = folder_name + r"\streamlines" + n + '_genu_cortex_cleaned.trk'
    streamlines_g = load_ft(tract_path)
    tract_path = folder_name + r"\streamlines" + n + '_body_cortex_cleaned.trk'
    streamlines_b = load_ft(tract_path)
    tract_path = folder_name + r"\streamlines" + n + '_splenium_cortex_cleaned.trk'
    streamlines_s = load_ft(tract_path)

    r = window.Renderer()
    genu_actor = actor.line(streamlines_g, (0,0.5,1), linewidth=0.5)
    r.add(genu_actor)

    body_actor = actor.line(streamlines_b, (0.5,1,0.5), linewidth=0.5)
    r.add(body_actor)

    splenium_actor = actor.line(streamlines_s, (1,0.5,0), linewidth=0.5)
    r.add(splenium_actor)
    window.show(r)
    save_as = folder_name + '\streamlines\cc_parts.png'
    r.set_camera(r.camera_info())
    window.record(r, out_path=save_as, size=(800, 800))



if __name__ == '__main__':
    subj = all_subj_folders
    names = all_subj_names
    masks = ['_genu_cortex_cleaned','_body_cortex_cleaned','_splenium_cortex_cleaned']
    weight_by = '1.5_2_AxPasi5'
    mean_vals = np.zeros([len(names),len(masks)])
    for s,n,i in zip(subj,names,range(len(names))):
        folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s
        dir_name = folder_name + '\streamlines'
        gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
        weight_by_file = bvec_file[:-5:] + '_' + weight_by + '.nii'
        weight_by_img = nib.load(weight_by_file)
        weight_by_data = weight_by_img.get_data()
        affine = weight_by_img.affine
        pfr_file = bvec_file[:-5:] + '_1.5_2_AxFr5.nii'
        pfr_img = nib.load(pfr_file)
        pfr_data = pfr_img.get_data()
        index_to_text_file = r'C:\Users\Admin\my_scripts\aal\megaatlas\megaatlascortex2nii.txt'
        for j,m in enumerate(masks):
            tract_path = dir_name + n + m + '.trk'
            streamlines = load_ft(tract_path)
            stream = list(streamlines)
            vol_per_tract = values_from_volume(weight_by_data, stream, affine=affine)
            pfr_per_tract = values_from_volume(pfr_data, stream, affine=affine)
            vol_vec = weight_by_data.flatten()
            q = np.quantile(vol_vec[vol_vec > 0], 0.95)
            mean_vol_per_tract = []
            for v, pfr in zip(vol_per_tract, pfr_per_tract):
                v = np.asanyarray(v)
                non_out = [v < q]
                pfr = np.asanyarray(pfr)
                high_pfr = [pfr > 0.5]
                mean_vol_per_tract.append(np.nanmean(v[tuple(non_out and high_pfr)]))

            mean_vol_per_part = np.nanmean(mean_vol_per_tract)
            mean_vals[i,j] = mean_vol_per_part
    np.save(r'C:\Users\Admin\my_scripts\Ax3D_Pack\mean_vals\different_way.npy',mean_vals)
    a=mean_vals
    plt.boxplot(a, widths=0.3, labels=['Genu', 'Body', 'Splenium'])
    plt.title('Mean AxCaliber (Pasi) values for CC parts')
    plt.show()

