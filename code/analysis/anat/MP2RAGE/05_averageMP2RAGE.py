"""Average images."""

import os
import numpy as np
import nibabel as nb
import glob

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
SUBS = ['sub-05']
UPFACT = 4

for sub in SUBS:

    # Find uni images in all sessions
    images = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/ses-*/anat/upsample/*uni*_registered.nii'))
    image1 = f'{DATADIR}/derivatives/{sub}/ses-01/anat/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups{UPFACT}X.nii.gz'


    outDir = f'{ROOT}/derivatives/{sub}/anat'
    outName = f'{sub}_uni_part-mag_avg_MP2RAGE_brain_ups{UPFACT}X'

    # =============================================================================
    print("MP2RAGE Step 05: Average.")

    # Load first file
    nii = nb.load(image1)
    data = np.squeeze(nii.get_fdata())

    for i in images:
        nii = nb.load(i)
        data += np.squeeze(nii.get_fdata())
    data /= (len(images) + 1)

    # Save
    img = nb.Nifti1Image(data, affine=nii.affine, header=nii.header)
    nb.save(img, os.path.join(outDir, "{}.nii.gz".format(outName)))

    print(f'Finished with {sub}.')
