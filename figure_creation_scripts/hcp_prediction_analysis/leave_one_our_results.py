import pandas as pd
import matplotlib.pyplot as plt


leave_one_out_results = pd.DataFrame(columns=['all','no_vis','no_default','no_cont','no_sommot','no_dorsattn','no_inter','no_limbic','no_salventattn'], index=['Num','FA','ADD','Dist'])

#Total:
total = leave_one_out_results.copy()
total['all'] = [0.2, 0.17, 0.25]

total['Cont'] = [0.06,0.09,0.12]
total['no_default'] = [0.06,0.09,0.12,0.05]
total['no_sommot'] = [0.07,0.13,0.11,0.04]
total['no_limbic'] = [0.06,0.11,0.12,0.05]
total['no_inter'] = [0.07,0.10,0.11,0.04]
total['no_dorsattn'] = [0.05,0.12,0.12,0.05]
total['no_salventattn'] = [0.06,0.11,0.11,0.05]
total['no_vis'] = [0.08,0.11,0.11,0.04]
total['n']
total = total.transpose()
total = total/total.loc[['all']].values
total = total.drop('all')
total=total*100
plt.figure(figsize=(10,6))
plt.plot(total)
plt.legend(total.columns)
plt.ylabel('% R-squared')
plt.title('Total')
plt.show()

#Fluid:
fluid = leave_one_out_results.copy()
fluid['all'] = [0.07,0.11,0.13,0.05]
fluid['no_cont'] = [0.07,0.10,0.08,0.04]
fluid['no_default'] = [0.07,0.09,0.12,0.05]
fluid['no_sommot'] = [0.07,0.12,0.11,0.04]
fluid['no_limbic'] = [0.07,0.12,0.13,0.05]
fluid['no_inter'] = [0.07,0.10,0.11,0.04]
fluid['no_dorsattn'] = [0.07,0.10,0.11,0.03]
fluid['no_salventattn'] = [0.06,0.12,0.11,0.05]
fluid['no_vis'] = [0.06,0.10,0.12,0.04]
fluid = fluid.transpose()
fluid = fluid/fluid.loc[['all']].values
fluid = fluid.drop('all')
fluid=fluid*100
plt.figure(figsize=(10,6))
plt.plot(fluid)
plt.legend(fluid.columns)
plt.ylabel('% R-squared')
plt.title('Fluid')
plt.show()

#Crystal:
crystal = leave_one_out_results.copy()
crystal['all'] = [0.07,0.12,0.13,0.05]
crystal['no_cont'] = [0.07,0.11,0.13,0.04]
crystal['no_default'] = [0.06,0.10,0.16,0.05]
crystal['no_sommot'] = [0.08,0.12,0.14,0.05]
crystal['no_limbic'] = [0.08,0.12,0.15,0.04]
crystal['no_inter'] = [0.06,0.10,0.14,0.05]
crystal['no_dorsattn'] = [0.07,0.11,0.13,0.04]
crystal['no_salventattn'] = [0.08,0.12,0.14,0.05]
crystal['no_vis'] = [0.07,0.10,0.15,0.05]
crystal = crystal.transpose()
crystal = crystal/crystal.loc[['all']].values
crystal = crystal.drop('all')
crystal=crystal*100
plt.figure(figsize=(10,6))
plt.plot(crystal)
plt.legend(crystal.columns)
plt.ylabel('% R-squared')
plt.title('Crystal')
plt.show()