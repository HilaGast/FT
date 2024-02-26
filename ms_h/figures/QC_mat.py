import matplotlib.pyplot as plt
import os, glob
import numpy as np

def show_subject_matrices(subj_fol, atlas):
    subj = subj_fol.split(os.sep)[-3]
    cm_fol = os.path.join(subj_fol, 'cm')
    add_mat_name = f'{cm_fol}{os.sep}add_{atlas}_cm_ord.npy'
    dist_file_name = f'{cm_fol}{os.sep}EucDist_{atlas}_cm_ord.npy'
    num_file_name = f'{cm_fol}{os.sep}num_{atlas}_cm_ord.npy'
    time_file_name = f'{cm_fol}{os.sep}time_th3_EucSym_{atlas}_cm_ord.npy'
    title = f'{subj} ADD'
    show_mat(add_mat_name, title, 2, 10)
    title = f'{subj} DIST'
    show_mat(dist_file_name, title, 1, 100)
    title = f'{subj} NUM'
    show_mat(num_file_name, title, 0, 200)
    title = f'{subj} TIME'
    show_mat(time_file_name, title, 0, 250)

def show_mat(mat_name, title, vmin, vmax):
    mat = np.load(mat_name)
    plt.imshow(mat, cmap='jet', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.show()

def run_all():
    # main_fol = 'F:\Hila\TDI\siemens'
    main_fol = r'F:\Hila\TDI\TheBase4Ever'
    # exp = 'D45d13'
    # all_subj_fol = glob.glob(f'{main_fol}{os.sep}[T]*{os.sep}{exp}{os.sep}')
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}[0-9]*{os.sep}')
    atlas = 'yeo7_100'
    # print(f'{atlas} -   {exp}')
    for subj_fol in all_subj_fol:
        #show_subject_matrices(subj_fol, atlas)
        subj = subj_fol.split(os.sep)[-2]
        cm_fol = os.path.join(subj_fol, 'cm')
        time_file_name = f'{cm_fol}{os.sep}{atlas}_time_th3_EucDist_cm_ord.npy'
        title = f'{subj} TIME'
        show_mat(time_file_name, title, 0, 200)
        input()


if __name__ == '__main__':
    run_all()