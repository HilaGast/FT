from network_analysis.community_based_topology import *
import glob, os



def run_community_rep_single_mat(mat_file, gd_file, weight_by, atlas):

    communities_vec = sio.loadmat(gd_file)['ciufull']
    show_topology(mat_file, communities_vec, weight_by, atlas, eflag=False, dup=100, nodes_norm_by='deg', is_edge_norm=True)


if __name__== '__main__':

    main_folder = r'G:\data\V7\HCP'

    files = glob.glob('G:\data\V7\HCP\cm\communities\group_division_yeo7_*.mat')
    for gd_file in files:
        if not 'SC' in gd_file:# and not 'SPE' in gd_file:
            print('unrecognized matrix type')
            continue
        if 'bna' in gd_file:
            atlas = 'bna'
        elif 'yeo7_200' in gd_file:
            atlas = 'yeo7_200'
        else:
            print('No atlas detected')
            continue

        f_parts = gd_file.split(os.sep)[-1].split('.')[0].split('_')[2:]
        weight_by = f'{atlas}_{f_parts[-3]}_{f_parts[-2]}_{f_parts[-1]}'
        mat_file = f'{main_folder}{os.sep}cm{os.sep}average_{weight_by}.npy'
        run_community_rep_single_mat(mat_file, gd_file, weight_by, atlas)