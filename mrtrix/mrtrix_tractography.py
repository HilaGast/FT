import os, glob

from fsl.file_prep import os_path_2_fsl


def create_tissue_map(mprage_brain, fivett):
    '''step 6: create a tissue map
    you'll need: MPRAGE_brain, mprage2diff, brain'''
    sgeroot = ''
    cmd = fr'bash -lc "5ttgen freesurfer {mprage_brain} {fivett} -premasked -nocrop -f"'
    os.system(cmd)


def fiber_tracking(base, data, fivett_reg2diff, response_wm, response_gm, response_csf, brain_mask, bvals, bvecs, response, fod, algorithm, ntracts, pixdim, minlength, tracts, fod_wm, sifted_tracts, final_ntracts):

    # step 7: the actual tractography
    '''you'll need: data (either the original one from the eddy, the one with only b-1000 or the one that went through resizing. depending on what you need), brain_mask, bvals & bvecs, brain_mask.

    you'll get: response functions for each tissue. fod, wither for each tissue or for WM, depending on the data'''

    cmd = fr'bash -lc "dwi2response dhollander {data} {response_wm} {response_gm} {response_csf} -mask {brain_mask} -force -fslgrad {bvecs} {bvals}"'
    os.system(cmd)

    cmd = fr'bash -lc "dwi2fod {algorithm} {data} {response} {fod} -fslgrad {base}/T1w/Diffusion/bvecs {base}/T1w/Diffusion/bvals -force -mask {brain_mask} "'# For b-1000 only, use algorithm = csd and only the path for the wm response function. for multi-shell, use msmt_csd and the paths for GM, WM and CSF response functions, in this order, with a path for the output fod after each one.
    os.system(cmd)

    cmd = fr'bash -lc "tckgen -force -algorithm SD_STREAM -select {ntracts} -step {pixdim * 0.5} -minlength {minlength} -maxlength {minlength}  -seed_image {brain_mask} -act {fivett_reg2diff} -fslgrad {bvecs} {bvals} {data} {fod_wm} {tracts} "' # things you might want to change, depending on what you want to do: ntracts - how many tracts to extract; I use ~4000000, but you might be ok with less. step - the step size. for minlength and maxlength, the common use is 30-500, but since the brains are different sizes I normalize it so that the average brain has 30-500, and all other brains have different values based on this.
    os.system(cmd)

    cmd = fr'bash -lc "tcksift -force -act {fivett_reg2diff} -fd_scale_gm -term_number {final_ntracts} {tracts} {fod_wm} {sifted_tracts}"' # this step filteres out tracts based on anatomical priors and wm density. for final_ntracts I use ~40000.
    os.system(cmd)



if __name__ == '__main__':
    main_fol = r'F:\Hila\TDI\siemens'
    exp = 'D45d13'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}[C,T]*{os.sep}{exp}{os.sep}')
    for subj in all_subj_fol:
        if os.path.exists(subj + f'streamlines{os.sep}wb_ssmt_act_mrtrix.tck'):
           continue
        subj = os_path_2_fsl(subj)
        mprage_brain = subj + 'rMPRAGE_brain.nii'
        five_tissue_types = subj + '5tt.nii'
        create_tissue_map(mprage_brain, five_tissue_types)
