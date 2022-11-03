"""Maximum intensity projection over one dimension for a window."""

import os
import numpy as np
from nibabel import load, Nifti1Image, save

nii = load('/path/to/file.nii.gz')

w = 10  # window, total width is double of this number

data = (nii.get_fdata()).astype("int")
dims = data.shape

# -----------------------------------------------------------------------------
suffix = 'maip_{}_x'.format(w)
temp = np.zeros(dims)
for i in range(w, dims[0]-w):
    temp[i, :, :] = np.max(data[i-w:i+w, :, :], axis=0)

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
    temp[:, i, :] = np.max(data[:, i-w:i+w, :], axis=1)

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
    temp[:, :, i] = np.max(data[:, :, i-w:i+w], axis=2)

print('Saving z...')
img = Nifti1Image(temp, affine=nii.affine)
temp = None
basename, ext = nii.get_filename().split(os.extsep, 1)
out_name = '{}_{}.{}'.format(basename, suffix, ext)
save(img, out_name)

print('Finished.')
