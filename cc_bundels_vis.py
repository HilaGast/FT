
from FT.weighted_tracts import load_ft,load_dwi_files
from dipy.viz import window, actor
from FT.all_subj import all_subj_folders, all_subj_names
import os
if __name__ == '__main__':
    subj = all_subj_folders
    names = all_subj_names
    for s, n in zip(subj,names):
        folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep' + s
        gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(folder_name)
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


