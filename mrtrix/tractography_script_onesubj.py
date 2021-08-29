import os
# step 1: extract first volume for AP and PA diffusion files and merge them into one

cmd = fr'bash -lc "fslroi {AP_diffusion} {nodif_AP} 0 1"'
os.system(cmd)

cmd = fr'bash -lc "fslroi {PA_diffusion} {nodif_PA} 0 1"'
os.system(cmd)

cmd = fr'bash -lc "fslmerge -t {AP_PA_b0} {nodif_AP} {nodif_PA}"'
os.system(cmd)

# step 2: topup creates an important file to use for the eddy corrections.
'''you'll need: AP_PA_b0 -> from step 1, datain -> a txt file based on the instructions here: [https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/Faq](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/Faq)

you'll get: topup_AP_PA_b0, topup_AP_PA_b0_iout, topup_AP_PA_b0_fout -> this is the only one you don't actually need'''

cmd = fr'bash -lc "topup  --imain={AP_PA_b0} --datain={datain} --config=b02b0.cnf --out={topup_AP_PA_b0} --iout={topup_AP_PA_b0_iout} --fout={topup_AP_PA_b0_fout}"'
os.system(cmd)

# step 3: another smoothing algorithm. has to happen before eddy (otherwhise it models the correction from eddy as noise)
'''you'll need: topup_AP_PA_b0_iout, AP_diffusion -> from previous steps/raw data

you'll get: hifi_nodif -> for internal use. nodif_brain (will also create nodif_brain_mask), dif_AP_denoised -> for next steps'''

cmd = fr'bash -lc "fslmaths {topup_AP_PA_b0_iout} -Tmean {hifi_nodif} "'
os.system(cmd)

cmd = fr'bash -lc "bet {hifi_nodif} {nodif_brain} -m -f 0.2 "' # this extracts a brain image without the skull. the -m option makes a brain mask image with the same name and the suffix _mask
os.system(cmd)

cmd = fr'bash -lc "dwidenoise -force -mask {nodif_brain_mask} {AP_diffusion} {dif_AP_denoised}"'
os.system(cmd)

# step 4: the actual eddy corrections
'''you'll need: dif_AP_denoised, nodif_brain_mask, topup_AP_PA_b0 -> from previous steps. index -> a txt file with a column of "1"s the same length as the number of volumes. datain -> same as before. bvecs, bvals -> from raw data

you'll get: data -> this is the DWI file you'll work with from now on. you only need this, the brain and brain mask, and the bval+bvec files'''

cmd = fr'bash -lc "eddy_openmp --imain={dif_AP_denoised} --mask={nodif_brain_mask} --index={index} --acqp={datain} --bvecs={bvecs} --bvals={bvals} --fwhm=0 --topup={topup_AP_PA_b0} --flm=quadratic --out={data}"' # --data_is_shelled
os.system(cmd)

# step 5: registrations. this step registeres both the atlas and the MPRAGE to the DWI
cmd = fr'bash -lc "bet {MPRAGE} {MPRAGE_brain} -f 0.3 -m"'
os.system(cmd)

cmd = fr'bash -lc "flirt -in {brain} -ref {MPRAGE_brain}  -omat {diff2MPRAGE} -bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12"'#creates a diffusion to mprage affine matrix
os.system(cmd)

cmd = fr'bash -lc "convert_xfm -inverse -omat {MPRAGE2diff} {diff2MPRAGE}"' # inverts to create an mprage2diff
os.system(cmd)

cmd = fr'bash -lc "flirt -ref {MNI_template} -in {MPRAGE} -omat {t12MNI_l}"' # create a linear registration affine matrix from t1 to MNI
os.system(cmd)

cmd = fr'bash -lc "fnirt --in={MPRAGE} --aff={t12MNI_l} --ref={MNI_template} --cout={t12MNI_nl} --config=T1_2_MNI152_2mm"' # create nonlinear warp for t1 to MNI
os.system(cmd)

cmd = fr'bash -lc "invwarp --ref={MPRAGE} --out={MNI2MPRAGE} --warp={t12MNI_nl}"' #invert the warp to be MNI to MPRAGE
os.system(cmd)

cmd = fr'bash -lc "applywarp --ref={MPRAGE} --in={AAL150} --out={atlas} --warp={MNI2MPRAGE}  --interp=nn"'
os.system(cmd)

cmd = fr'bash -lc "flirt -applyxfm  -ref {brain} -in {atlas}  -init {MPRAGE2diff}  -interp nearestneighbour -out {atlas}"'
os.system(cmd)

# step 6: create a tissue map
'''SGE_ROOT='' 5ttgen fsl {MPRAGE_brain} {5tt} -premasked -nocrop -f
transformconvert -force {mprage2diff} {MPRAGE_brain} {brain} flirt_import {MPRAGE2diff.txt}
mrtransform {5tt} {5tt_reg2diff} -force -linear {MPRAGE2diff.txt}- **step 7**: create a tissue map. this step creates a file with 5 volumes, each representing the tissue types: GM, sub-cortical GM, WM, CSF, non-brain tissue. this is important for the tractography priors - to make the tracts based on anatomical data. we make it on the MPRAGE and then use the transform matrix from before for the registration to diffusion.

you'll need: MPRAGE_brain, mprage2diff, brain'''

cmd = fr'bash -lc "SGE_ROOT='' 5ttgen fsl {MPRAGE_brain} {5tt} -premasked -nocrop -f"'
os.system(cmd)

cmd = fr'bash -lc "transformconvert -force {mprage2diff} {MPRAGE_brain} {brain} flirt_import {MPRAGE2diff.txt}"'
os.system(cmd)

cmd = fr'bash -lc "mrtransform {5tt} {5tt_reg2diff} -force -linear {MPRAGE2diff.txt}"'
os.system(cmd)


# step 7: the actual atractography
'''you'll need: data (either the original one from the eddy, the one with only b-1000 or the one that went through resizing. depending on what you need), brain_mask, bvals & bvecs, brain_mask. 

you'll get: response functions for each tissue. fod, wither for each tissue or for WM, depending on the data'''

cmd = fr'bash -lc "dwi2response dhollander {data} {response_wm} {response_gm} {response_csf} -mask {brain_mask} -force -fslgrad {bvecs} {bvals}"'
os.system(cmd)

cmd = fr'bash -lc "dwi2fod {algorithm} {data} {response} {fod} -fslgrad {base}/T1w/Diffusion/bvecs {base}/T1w/Diffusion/bvals -force -mask {brain_mask} "'# For b-1000 only, use algorithm = csd and only the path for the wm response function. for multi-shell, use msmt_csd and the paths for GM, WM and CSF response functions, in this order, with a path for the output fod after each one.
os.system(cmd)

cmd = fr'bash -lc "tckgen -force -algorithm SD_STREAM -select {ntracts} -step {pixdim * 0.5} -minlength {minlength} -maxlength {minlength}  -seed_image {brain_mask} -act {5tt_reg2diff} -fslgrad {bvecs} {bvals} {data} {fod_wm} {tracts} "' # things you might want to change, depending on what you want to do: ntracts - how many tracts to extract; I use ~4000000, but you might be ok with less. step - the step size. for minlength and maxlength, the common use is 30-500, but since the brains are different sizes I normalize it so that the average brain has 30-500, and all other brains have different values based on this.
os.system(cmd)

cmd = fr'bash -lc "tcksift -force -act {5tt_reg2diff} -fd_scale_gm -term_number {final_ntracts} {tracts} {fod_wm} {sifted_tracts}"' # this step filteres out tracts based on anatomical priors and wm density. for final_ntracts I use ~40000.
os.system(cmd)