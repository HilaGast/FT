
from FT.weighted_tracts import load_ft,load_dwi_files
from dipy.viz import window, actor
from FT.all_subj import all_subj_names

if __name__ == '__main__':
    subj = all_subj_names[0:1]
    for s in subj:
        folder_name = r'C:\Users\Admin\my_scripts\Ax3D_Pack\V5' + s
        gtab, data, affine, labels, white_matter, nii_file = load_dwi_files(folder_name)
        tract_path = folder_name + r"\streamlines" + s + '_genu_cortex.trk'
        streamlines_g = load_ft(tract_path)
        tract_path = folder_name + r"\streamlines" + s + '_body_cortex.trk'
        streamlines_b = load_ft(tract_path)
        tract_path = folder_name + r"\streamlines" + s + '_splenium_cortex.trk'
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


