import glob, os
import numpy as np
from bct.algorithms.similarity import corr_flat_und
from Tractography.group_analysis import create_all_subject_connectivity_matrices, norm_matrices
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pingouin import rm_anova
from basic_statistics import ttest

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}')
atlas = 'yeo7_100'
hemi = 'LH'
s_mat_types = ['Dist', 'FA', 'ADD', 'time_th3']
all_r = dict()
for mat_type in s_mat_types:
    structure_names = []
    fmri_names = []
    for s in subj_list:
        structure_file_name = f'{s}cm{os.sep}{atlas}_{mat_type}_Org_SC_cm_ord.npy'
        fmri_file_name = f'{s}cm{os.sep}{atlas}_fmri_Org_SC_cm_ord.npy'
        if os.path.exists(structure_file_name) and os.path.exists(fmri_file_name):
            structure_names.append(structure_file_name)
            fmri_names.append(fmri_file_name)
    # Load
    structure_mats = create_all_subject_connectivity_matrices(structure_names)
    fmri_mats = create_all_subject_connectivity_matrices(fmri_names)

    # choose hemisphere:
    if hemi == 'RH':
        structure_mats = structure_mats[50::, 50::, :]
        fmri_mats = fmri_mats[50::, 50::, :]
    elif hemi == 'LH':
        structure_mats = structure_mats[:50, :50, :]
        fmri_mats = fmri_mats[:50, :50, :]

    # revert tdi?
    if 'time' in mat_type:
        structure_max_vals = np.nanmax(structure_mats, axis=(0, 1))
        structure_mats = structure_max_vals - structure_mats
        mat_type = 'TDI'

    # abs fmri?
    fmri_mats = np.abs(fmri_mats)

    # Normalize
    structure_norm = norm_matrices(structure_mats, norm_type='scaling')
    fmri_norm = norm_matrices(fmri_mats, norm_type='scaling')

    # remove Nans:
    structure_norm[np.isnan(structure_norm)] = 0
    fmri_norm[np.isnan(fmri_norm)] = 0

    # Find distance\similarity between tdi and fmri
    r = []
    for i in range(structure_norm.shape[-1]):
        ri = corr_flat_und(structure_norm[:, :, i], fmri_norm[:, :, i])
        r.append(ri)

    all_r[mat_type] = r

df_org = pd.DataFrame.from_dict(all_r)
df = df_org.melt(var_name='Weighted SC', value_name='SC-FC similarity (r)')
#df = df.sort_values(by=['Weighted SC'])
plt.figure(figsize=(6.5, 6))
sns.set_theme(style="whitegrid", palette="pastel", font_scale=1.2)
# ax = sns.boxplot(x="Network", y="Scaled TDI [0-1]", data=df)
ax = sns.boxplot(x="Weighted SC", y="SC-FC similarity (r)", data=df, width=0.5)
plt.title(f'{hemi}')
plt.show()


# ANOVA
aov = rm_anova(data=df_org)
print(f"F={aov['F'][0]},p={aov['p-unc'][0]}")
print(aov)

# Post-hoc
t, p = ttest(df_org['TDI'],df_org['FA'], type='related', alternative='two-sided')
print(f'FA: t={t}, p={p}    after Bonferroni correction: {p/3}')
t, p = ttest(df_org['TDI'],df_org['ADD'], type='related', alternative='two-sided')
print(f'ADD: t={t}, p={p}    after Bonferroni correction: {p/3}')
t, p = ttest(df_org['TDI'],df_org['Dist'], type='related', alternative='two-sided')
print(f'Dist: t={t}, p={p}    after Bonferroni correction: {p/3}')










