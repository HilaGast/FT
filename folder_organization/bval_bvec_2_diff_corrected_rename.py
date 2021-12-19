import glob,os
def change_name(main_subj_folder,rename_dest):
    for bv_file in glob.glob(f'{main_subj_folder}{os.sep}*{os.sep}*.bv*'):
        file_name_parts= bv_file.split(os.sep)
        if 'APs' in file_name_parts[-1] and file_name_parts[-1].endswith('.bval'):
            new_file_name = f'{rename_dest}.bval'
        elif 'APs' in file_name_parts[-1] and file_name_parts[-1].endswith('.bvec'):
            new_file_name = f'{rename_dest}.bvec'
        else:
            new_file_name = file_name_parts[-1]

        file_name_parts[-1] = new_file_name
        new_file_path = os.path.join(*file_name_parts)
        os.rename(bv_file,new_file_path)

if __name__ == '__main__':
    fol = r'Y:\data\subj\thebase4ever'

    change_name(fol,'diff_corrected')