from all_subj import *
import nibabel as nib


i=8
s = all_subj_folders[i]
n = all_subj_names[i]

folder_name = subj_folder + s
for file in os.listdir(folder_name):
    if file.startswith("r") and file.endswith("001_brain.nii"):
        mprage_file = os.path.join(folder_name, file)
    if file.endswith("brain_seg.nii"):
        labels_file_name = os.path.join(folder_name, file)

labels_img = nib.load(labels_file_name)
labels = labels_img.get_fdata()
white_matter = (labels == 3) #| (labels == 2)  # 3-WM, 2-GM
non_gm = (labels != 2)

mprage_img = nib.load(mprage_file)
mprage = mprage_img.get_fdata()


masked_mprage = white_matter*mprage
masked_file = nib.Nifti1Image(masked_mprage,mprage_img.affine,mprage_img.header)
nib.save(masked_file,labels_file_name[:-4]+"_WMmasked.nii")

masked_mprage = non_gm*mprage
masked_file = nib.Nifti1Image(masked_mprage,mprage_img.affine,mprage_img.header)
nib.save(masked_file,labels_file_name[:-4]+"_nonGMmasked.nii")

