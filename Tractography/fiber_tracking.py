from dipy.io.image import load_nifti, load_nifti_data
from Tractography.files_loading import *



class Tractography():
    def __init__(self, subj_main_folder, ft_method, sc_method, seed_type, parameters_dict, diff_file_name = 'data', mask_type=''):
        self.subj_folder = subj_main_folder
        self.sc_method = sc_method
        self.ft_method = ft_method
        self.seed_type = seed_type
        self.mask_type = mask_type
        self.parameters_dict = parameters_dict
        self.three_tissue_labels = load_pve_files(self.subj_folder)[3] # wm=3,gm=2, csf=1
        self.gtab = bval_bvec_2_gtab(self.subj_folder, small_delta=15.5)
        self.nii_ref, self.data, self.affine = load_nii_file(self.subj_folder,diff_file_name)
        voxel_size = np.mean(load_nifti(self.nii_ref,return_voxsize=True)[2])
        self.parameters_dict['voxel_size'] = voxel_size
        self.parameters_dict['length_margins'] = self.parameters_dict['length_margins_mm']/(self.parameters_dict['step_size']*self.parameters_dict['voxel_size'])
        self.create_seeds()

    def create_seeds(self):
        from dipy.tracking.utils import seeds_from_mask

        if self.seed_type == 'gm':
            seed_mask = self.tissue_labels = 2

        elif self.seed_type == 'mask':
            mask_mat = load_mask(self.subj_folder, self.mask_type)
            seed_mask = mask_mat == 1

        elif self.seed_type == 'wm':
            seed_mask = self.tissue_labels = 3

        elif self.seed_type == 'wb':
            seed_mask = self.tissue_labels > 0

        else:
            print("Couldn't recognize seed type, please specify one of the following: gm, wm, mask, wb")

        seeds = seeds_from_mask(seed_mask, density=self.parameters_dict['den'], affine=self.affine)

        self.seeds = seeds

    def create_model_fit(self):
        if self.ft_method == 'csd':
            self._csd_ft()
        elif self.ft_method == 'msmt':
            self._msmt_ft()

    def _csd_ft(self):
        from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst

        response, ratio = auto_response_ssst(self.gtab, self.data, roi_radii=10, fa_thr=0.7)
        csd_model = ConstrainedSphericalDeconvModel(self.gtab, response, sh_order=self.parameters_dict['sh_order'])
        csd_fit = csd_model.fit(self.data)
        self.model_fit = csd_fit

    def _msmt_ft(self):
        from dipy.reconst.mcsd import response_from_mask_msmt
        from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response, MSDeconvFit
        from dipy.core.gradients import unique_bvals_tolerance

        bvals = self.gtab.bvals
        wm = self.tissue_labels == 3
        gm = self.tissue_labels == 2
        csf = self.tissue_labels == 1

        mask_wm = wm.astype(float)
        mask_gm = gm.astype(float)
        mask_csf = csf.astype(float)

        response_wm, response_gm, response_csf = response_from_mask_msmt(self.gtab, self.data,
                                                                         mask_wm,
                                                                         mask_gm,
                                                                         mask_csf)

        ubvals = unique_bvals_tolerance(bvals)
        response_mcsd = multi_shell_fiber_response(self.parameters_dict['sh_order'], bvals=ubvals,
                                                   wm_rf=response_wm,
                                                   csf_rf=response_csf,
                                                   gm_rf=response_gm)
        mcsd_model = MultiShellDeconvModel(self.gtab, response_mcsd)

        mcsd_fit = mcsd_model.fit(self.data)
        sh_coeff = mcsd_fit.all_shm_coeff
        nan_count = len(np.argwhere(np.isnan(sh_coeff[..., 0])))
        coeff = mcsd_fit.all_shm_coeff
        n_vox = coeff.shape[0] * coeff.shape[1] * coeff.shape[2]
        if nan_count > 0:
            print(f'{nan_count / n_vox} of the voxels did not complete fodf calculation, NaN values replaced with 0')
        coeff = np.where(np.isnan(coeff), 0, coeff)
        mcsd_fit = MSDeconvFit(mcsd_model, coeff, None)
        self.model_fit = mcsd_fit

    def create_classifier(self):
        if self.sc_method == 'fa':
            self._fa_sc()
        elif self.sc_method == 'act':
            self._act_sc()
        elif self.sc_method == 'cmc':
            self._cmc_sc()

    def _fa_sc(self):
        import dipy.reconst.dti as dti
        from dipy.reconst.dti import fractional_anisotropy
        from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

        tensor_model = dti.TensorModel(self.gtab)
        tenfit = tensor_model.fit(self.data, mask=self.tissue_labels == 3)
        fa = fractional_anisotropy(tenfit.evals)
        classifier = ThresholdStoppingCriterion(fa, self.parameters_dict['fa_th'])
        self.classifier = classifier

    def _act_sc(self):
        from dipy.tracking.stopping_criterion import ActStoppingCriterion

        act_classifier = ActStoppingCriterion()
        self.classifier = act_classifier

    def _cmc_sc(self):
        from dipy.tracking.stopping_criterion import CmcStoppingCriterion

        f_pve_csf, f_pve_gm, f_pve_wm = load_pve_files(self.subj_folder)
        pve_csf_data = load_nifti_data(f_pve_csf)
        pve_gm_data = load_nifti_data(f_pve_gm)
        pve_wm_data = load_nifti_data(f_pve_wm)
        cmc_criterion = CmcStoppingCriterion.from_pve(pve_wm_data,
                                                      pve_gm_data,
                                                      pve_csf_data,
                                                      step_size=self.parameters_dict['step_size'],
                                                      average_voxel_size=self.parameters_dict['voxel_size'])
        self.classifier = cmc_criterion

    def _remove_streamlines_outliers(self):
        keep_streamlines = np.ones((len(self.streamlines)), bool)
        for i in range(0, len(self.streamlines)):
            if self.streamlines[i].shape[0] < self.parameters_dict['length_margins'][0]:
                keep_streamlines[i] = False
            elif self.streamlines[i].shape[0] > self.parameters_dict['length_margins'][1]:
                keep_streamlines[i] = False

        self.streamlines = self.streamlines[keep_streamlines]

    def fiber_tracking(self):
        from dipy.data import default_sphere
        from dipy.direction import DeterministicMaximumDirectionGetter
        from dipy.tracking.streamline import Streamlines
        from dipy.tracking.local_tracking import ParticleFilteringTracking
        from Tractography.files_saving import save_ft

        self.create_model_fit()
        detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(self.model_fit.shm_coeff,
                                                                     max_angle=self.parameters_dict['max_ang'],
                                                                     sphere=default_sphere)
        print(f"Tractography using PFT and {self.sc_method} clasifier")
        self.create_classifier()

        print('Starting to compute streamlines')
        self.streamlines = Streamlines(ParticleFilteringTracking(
            detmax_dg, self.classifier, self.seeds, self.affine, step_size=self.parameters_dict['step_size'],
            maxlen=self.parameters_dict['length_margins'][1],
            pft_back_tracking_dist=2,
            pft_front_tracking_dist=1,
            particle_count=15,
            return_all=False))

        self._remove_streamlines_outliers()

        file_name = f'wb_{self.ft_method}_{self.sc_method}.tck'
        save_ft(self.subj_folder, self.streamlines, self.nii_ref, file_name)


def fiber_tracking_parameters(max_angle = 30, sh_order = 8, seed_density = 4, streamlines_lengths_mm = [30,500], step_size = 0.2, fa_th = 0.18):
    parameters_dict = dict()
    parameters_dict['max_angle'] = max_angle
    parameters_dict['sh_order'] = sh_order
    parameters_dict['den'] = seed_density
    parameters_dict['length_margins_mm'] = streamlines_lengths_mm
    parameters_dict['step_size'] = step_size  # 0.2 for msmt, 1 for csd
    parameters_dict['fa_th'] = fa_th
    return parameters_dict



