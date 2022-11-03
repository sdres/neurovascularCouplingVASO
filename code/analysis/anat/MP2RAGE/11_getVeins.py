import numpy as np
import nibabel as nb

file = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/veinsPlusPeri.nii.gz'
nii = nb.load(file)
data = nii.get_fdata()
np.unique(data)


trueveins = np.zeros(data.shape)

trueveins = np.where(data == 1, 1,0)
gmVeins = np.where(data == 11, 1,0)


trueveins += gmVeins

img = nb.Nifti1Image(trueveins, header = nii.header, affine = nii.affine)
nb.save(img, '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/finalveins.nii.gz')
