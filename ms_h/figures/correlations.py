import glob,os
import pandas as pd
from scipy.stats import pearsonr
from draw_scatter_fit import remove_nans,draw_scatter_fit

if __name__ == '__main__':
    main_fol = 'Y:\qnap\siemens'
    all_subj_fol = glob.glob(f'{main_fol}{os.sep}*{os.sep}AxCaliber{os.sep}')
    all_subj_names = [s.split(os.sep)[3] for s in all_subj_fol]
    vol_names = ['FA', 'MD', 'ADD', 'pFr', 'pH', 'pCSF']
    experiments = ['D60d11', 'D45d13', 'D31d18']
    cc_parts = ["Total","G","AB","MB","PB","Sp"] #CC

    parts = ["WB", "CC"]
    correlation_table = pd.DataFrame(columns=['scan_type','part','var1','var2','r','p'])
    multi_index = pd.MultiIndex.from_product([experiments, vol_names], names=["Experiment", "Volume"])  # WB
    table_wb = pd.read_excel(r'Y:\qnap\siemens_results_WB\results.xlsx','raw',names=multi_index)
    table_years = pd.read_excel(r'Y:\qnap\siemens\age_and_duration.xlsx')
    multi_index = pd.MultiIndex.from_product([['CC vol']+vol_names, cc_parts], names=["Volume","Parts"]) #CC
    for experiment in experiments:
        table_cc = pd.read_excel(fr'Y:\qnap\siemens_results_CC\results_{experiment}_new.xlsx','raw', names=multi_index)
        for part in parts:
            for vol in vol_names[:3]:
                x1=[]#Age
                x2=[]#DD
                y=[]
                for subj in all_subj_names:
                    x1.append(table_years['Age'][table_years['subj']==subj].values[0])
                    x2.append(table_years['Disease duration'][table_years['subj']==subj].values[0])
                    if part=='WB':
                        y.append(table_wb[experiment,vol][subj])
                    else:
                        y.append(table_cc[vol,'Total'][subj])
                x1, y1 = remove_nans(x1,y)
                r, p = pearsonr(x1, y1)
                correlation_table.loc[len(correlation_table.index)] = [experiment, part, 'Age', vol, r, p]
                #draw_scatter_fit(x1,y1,deg=1,comp_reg=True,ttl=f'{experiment} - {part} \n Age - {vol}', norm_x=False)

                x2,y2 = remove_nans(x2,y)
                r, p = pearsonr(x2, y2)
                correlation_table.loc[len(correlation_table.index)] = [experiment, part, 'DD', vol, r, p]
                draw_scatter_fit(x2,y2,deg=1,comp_reg=True,ttl=f'{experiment} - {part} \n Disease Duration - {vol}', norm_x=False)

    #correlation_table.to_excel(f'Y:\qnap\siemens{os.sep}correlation_with_Age_and_DD.xlsx')
