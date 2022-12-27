import os,glob

subj_list = glob.glob(f'G:\data\V7\HCP\*[0-9]{os.sep}\cm\*.npy')

for sl in subj_list:
    if 'lookup' in sl or 'unsifted' in sl:
        continue
    else:
        fol = os.path.dirname(sl)

        if 'bna' in sl:
            atlas = 'bna'
        elif 'yeo7_200' in sl:
            atlas = 'yeo7_200'
        elif 'mega' in sl:
            atlas = 'mega'
        else:
            print (f"Couldn't recognize atlas for {sl}")
            continue

        if 'NumxDist' in sl:
            w = 'NumxDist'
        elif 'NumxADD' in sl:
            w = 'NumxADD'
        elif 'ADDxDist' in sl:
            w = 'ADDxDist'
        elif 'add' in sl and 'num' not in sl:
            w = 'ADD'
        elif 'fa' in sl and 'num' not in sl:
            w = 'FA'
        elif 'num' in sl and 'add' not in sl and 'fa' not in sl:
            w = 'Num'
        elif 'dist' in sl:
            w = 'Dist'
        else:
            print(f"Couldn't recognize weight for {sl}")
            continue

        if 'histmatch' in sl:
            rescale = 'HistMatch'
        else:
            rescale = 'Org'

        if 'SPE' in sl:
            ncm = 'SPE'
        elif 'NE' in sl:
            ncm = 'NE'
        elif 'CMY' in sl:
            ncm = 'CMY'
        else:
            ncm = 'SC'

        new_name = f'{fol}{os.sep}{atlas}_{w}_{rescale}_{ncm}_cm_ord.npy'

        os.rename(sl,new_name)
