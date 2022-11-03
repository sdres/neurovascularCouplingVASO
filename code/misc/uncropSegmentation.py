import nibabel as nb
import numpy as np

anatFolder = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/anat/upsample'
gmFile = f'{anatFolder}/peri_layers.nii.gz'
anatFile = f'{anatFolder}/sub-06_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'


anat = nb.load(anatFile)
header = anat.header
affine = anat.affine
anatData = anat.get_fdata()

gm = nb.load(gmFile)
# header = gm.header
# affine = gm.affine
gmData = gm.get_fdata()
gmData.shape

new = np.zeros(anatData.shape)
new[271:433,7:169,31:190] = gmData
# new[263:425,79:238,35:197] = gmData
# new[35:197,263:425,79:238] = gmData
# new[35:197,79:238,263:425] = gmData
# new[79:238,35:197,263:425] = gmData
# new[79:238,263:425,35:197] = gmData
# '271 162 7 162 31 159'
img = nb.Nifti1Image(new, header=header,affine=affine)
nb.save(img, f'{anatFolder}/periLayers_uncrop.nii.gz')
