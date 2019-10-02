import os
fdwi= 'C:/Users/Admin/my_scripts/C1156_43/YA_lab_Assi_GaHi_20171231_1320/23_AxCaliber3D1_ep2d_advdiff_30dir_b3000_d11.3D60/20171231_132011AxCaliber3D1ep2dadvdiff30dirb3000d113D60s023a001.nii'
fbvec= 'C:/Users/Admin/my_scripts/C1156_43/YA_lab_Assi_GaHi_20171231_1320/23_AxCaliber3D1_ep2d_advdiff_30dir_b3000_d11.3D60/20171231_132011AxCaliber3D1ep2dadvdiff30dirb3000d113D60s023a001.bvec'
fbval='C:/Users/Admin/my_scripts/C1156_43/YA_lab_Assi_GaHi_20171231_1320/23_AxCaliber3D1_ep2d_advdiff_30dir_b3000_d11.3D60/20171231_132011AxCaliber3D1ep2dadvdiff30dirb3000d113D60s023a001.bval'

from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

data, affine = load_nifti(fdwi)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data)

save_nifti('C:/Users/Admin/my_scripts/C1156_43/YA_lab_Assi_GaHi_20171231_1320/23_AxCaliber3D1_ep2d_advdiff_30dir_b3000_d11.3D60/colorfa.nii.gz', tenfit.color_fa, affine)

