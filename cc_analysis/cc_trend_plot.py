from cc_analysis.cc_boxplot import *

def show_cc_trend_plot(table):
    import matplotlib.pyplot as plt
    val_list = list(table['ADD'])
    subj_cc=np.zeros((5,int(len(val_list)/5)))
    for i, v in enumerate(val_list[0:len(val_list):5]):
        cc = val_list[i * 5:i * 5 + 5]
        subj_cc[:,i] = cc
    plt.plot(subj_cc,'*-')
    plt.show()







if __name__ == '__main__':

    main_path = r'F:\Hila\Ax3D_Pack\V6\v7calibration\Old_AxCaliber\H'
    cc_parts_table_H = compare_deltas_old_axcaliber(main_path, group='H')
    main_path = r'F:\Hila\Ax3D_Pack\V6\v7calibration\Old_AxCaliber\MS'
    cc_parts_table_MS = compare_deltas_old_axcaliber(main_path, group='MS')

    cc_parts_table = cc_parts_table_H.append(cc_parts_table_MS)
    protocols = ['D31 d18', 'D45 d13.2', 'D60 d11.3']
    show_cc_trend_plot(cc_parts_table_H[cc_parts_table_H['Protocol']==protocols[2]])