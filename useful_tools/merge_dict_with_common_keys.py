import numpy as np
def merge_dict_concatenate(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = np.concatenate((value , dict_1[key]), axis=1)
   return dict_3

def merge_dict_sum(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = np.nansum((value , dict_1[key]))
   return dict_3

