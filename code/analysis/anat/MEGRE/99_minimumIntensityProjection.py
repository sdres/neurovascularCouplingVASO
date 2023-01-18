"""Maximum intensity projection over one dimension for a window."""

import os
import numpy as np
from nibabel import load, Nifti1Image, save

nii = load('/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/11_T2star/sub-06_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_T2s_registered_crop-toShpereLH.nii.gz')

w = 2  # window, total width is double of this number

data = (nii.get_fdata()).astype("int")
dims = data.shape

# -----------------------------------------------------------------------------
suffix = 'maip_{}_x'.format(w)
temp = np.zeros(dims)
for i in range(w, dims[0]-w):
    temp[i, :, :] = np.min(data[i-w:i+w, :, :], axis=0)

print('Saving x...')
img = Nifti1Image(temp, affine=nii.affine)
temp = None
basename, ext = nii.get_filename().split(os.extsep, 1)
out_name = '{}_{}.{}'.format(basename, suffix, ext)
save(img, out_name)

# -----------------------------------------------------------------------------
suffix = 'maip_{}_y'.format(w)
temp = np.zeros(dims)
for i in range(w, dims[1]-w):
    temp[:, i, :] = np.min(data[:, i-w:i+w, :], axis=1)

print('Saving y...')
img = Nifti1Image(temp, affine=nii.affine)
temp = None
basename, ext = nii.get_filename().split(os.extsep, 1)
out_name = '{}_{}.{}'.format(basename, suffix, ext)
save(img, out_name)

# -----------------------------------------------------------------------------
suffix = 'maip_{}_z'.format(w)
temp = np.zeros(dims)
for i in range(w, dims[2]-w):
    temp[:, :, i] = np.min(data[:, :, i-w:i+w], axis=2)

print('Saving z...')
img = Nifti1Image(temp, affine=nii.affine)
temp = None
basename, ext = nii.get_filename().split(os.extsep, 1)
out_name = '{}_{}.{}'.format(basename, suffix, ext)
save(img, out_name)

print('Finished.')
