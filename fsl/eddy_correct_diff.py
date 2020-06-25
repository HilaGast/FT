import os

def eddy_corr(subj_folder,diff_file_name, pa_file_name, acqr_file = r'/mnt/c/Users/Admin/my_scripts/Ax3D_Pack/V5/datain.txt', index_file = r'/mnt/c/Users/Admin/my_scripts/Ax3D_Pack/V5/index.txt'):
    cmd = fr'bash -lc "fslroi {subj_folder}/{diff_file_name} {subj_folder}/AP_nodiff 0 1"'
    os.system(cmd)
    cmd = fr'bash -lc "fslroi {subj_folder}/{pa_file_name} {subj_folder}/PA_nodiff 0 1"'
    os.system(cmd)
    cmd = fr'bash -lc "fslmerge -t {subj_folder}/AP_PA_b0.nii {subj_folder}/AP_nodiff.nii {subj_folder}/PA_nodiff.nii"'
    os.system(cmd)
    cmd = fr'bash -lc "topup --imain={subj_folder}/AP_PA_b0.nii --datain={acqr_file} --config=b02b0.cnf --out={subj_folder}/topup_AP_PA_b0 --iout={subj_folder}/topup_AP_PA_b0_iout --fout={subj_folder}/topup_AP_PA_b0_fout"'
    os.system(cmd)
    cmd = fr'bash -lc "fslmaths {subj_folder}/topup_AP_PA_b0_iout -Tmean {subj_folder}/hifi_nodif"'
    os.system(cmd)
    cmd = fr'bash -lc "bet {subj_folder}/hifi_nodif {subj_folder}/hifi_nodif_brain -m -f 0.3 -g 0.2"'
    os.system(cmd)
    cmd = fr'bash -lc "eddy_openmp --imain={subj_folder}/{diff_file_name} --mask={subj_folder}/hifi_nodif_brain_mask --index={index_file} --acqp={acqr_file} --bvecs={subj_folder}/{diff_file_name[:-4]}.bvec --bvals={subj_folder}/{diff_file_name[:-4]}.bval --fwhm=0 --topup={subj_folder}/topup_AP_PA_b0 --flm=quadratic --out={subj_folder}/diff_corrected.nii --data_is_shelled"'
    os.system(cmd)
