import os

def save_ft(folder_name, streamlines, nii_file, file_name = "wholebrain.trk"):
    from dipy.io.streamline import save_trk, save_tck
    from dipy.io.stateful_tractogram import StatefulTractogram, Space

    dir_name = f'{folder_name}{os.sep}streamlines'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    tract_name = dir_name + os.sep + file_name

    if tract_name.endswith('.trk'):
        save_trk(StatefulTractogram(streamlines,nii_file,Space.RASMM),tract_name)
    elif tract_name.endswith('.tck'):
        save_tck(StatefulTractogram(streamlines,nii_file,Space.RASMM),tract_name)
