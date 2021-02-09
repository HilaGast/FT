from weighted_tracts import *
import matplotlib.pyplot as plt


def find_non_zeros(s,matname1,matname2):
    l1=[]
    l2=[]
    mat1 = np.load(rf'{subj_folder}{s}\{matname1}.npy')
    mat2 = np.load(rf'{subj_folder}{s}\{matname2}.npy')

    for col in range(len(mat1)):
        for row in range(col + 1):
            if mat1[col,row] != 0:
                l1+=[mat1[col,row]]
                l2+=[mat2[col,row]]
    if np.mean(l1)<6:
        print(rf'{s}:   {np.mean(l1)}')


    return l1,l2


def cluster(group_by_array, dragged_array, maxdiff):
    tmp1 = group_by_array.copy()
    tmp2 = dragged_array.copy()
    groups = []
    while len(tmp1):
        # select seed
        seed = tmp1.min()
        mask = (tmp1 - seed) <= maxdiff
        groups.append(tmp2[mask, None])
        tmp1 = tmp1[~mask]
        tmp2 = tmp2[~mask]

    return groups

if __name__ == '__main__':
    subj = all_subj_folders
    matname1 = 'weighted_wholebrain_4d_labmask_yeo7_200_nonnorm'
    matname2 = 'weighted_yeo7_lengths_nonnorm'
    list1=[]
    list2=[]
    for s in subj:
        l1,l2 = find_non_zeros(s,matname1,matname2)
        list1+=l1
        list2+=l2
    list1 = np.asarray(list1)
    list2 = np.asarray(list2)
    del_nan = np.isnan(list1)
    list1 = list1[~del_nan]
    list2 = list2[~del_nan]
    plt.scatter(list2,list1,5,'green')
    plt.xlabel('Average axon length (mm)')
    plt.ylabel('Average axon diameter (mm)')
    plt.show()

    groups = cluster(list2,list1,50)
    plt.boxplot(groups,meanline=True,showmeans=True)
    plt.show()