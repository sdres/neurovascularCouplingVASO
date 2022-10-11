"""Average images."""

import os
import numpy as np
import nibabel as nb
import glob
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
SUBS = ['sub-05']

for sub in SUBS:
    inDir = f'{ROOT}/derivatives/{sub}/anat/upsample'
    outDir = f'{ROOT}/derivatives/{sub}/anat/upsample/registration'


    image1 = f'{inDir}/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_brain_ups2X.nii.gz'

    images = sorted(glob.glob(f'{outDir}/*uni*.nii'))

    OUT_NAME = f'{inDir}/{sub}_ses-01_uni_part-mag_avg_MP2RAGE_brain_ups2X'

    # =============================================================================
    print("MP2RAGE Step 05: Average.")

    # Load first file
    nii = nb.load(image1)
    data = np.squeeze(nii.get_fdata()) * 0

    for i in images:
        nii = nb.load(i)
        data += np.squeeze(nii.get_fdata())
    print("  Nr nifti files:{}".format(len(images)+1))
    data /= len(images)+1

    # Save
    img = nb.Nifti1Image(data, affine=nii.affine, header=nii.header)
    nb.save(img, os.path.join(inDir, "{}.nii.gz".format(OUT_NAME)))

    print('Finished.')
