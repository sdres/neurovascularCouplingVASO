'''

Apply transformation to uni images.

'''

import ants
import os
import glob
import subprocess

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

subs = ['sub-05']

for sub in subs:

    inDir = f'{ROOT}/derivatives/{sub}/anat/upsample'
    outDir = f'{ROOT}/derivatives/{sub}/anat/upsample/registration'


    # Find brain extracted inv-2 images
    images = sorted(glob.glob(f'{inDir}/*uni*brain_up*.nii.gz'))
    transforms = sorted(glob.glob(f'{outDir}/*transform.mat'))

    fixed = ants.image_read(images[0])
    # =========================================================================
    # Apply transform

    for image, transform in zip(images[1:],transforms):
        base = os.path.basename(image).rsplit('.', 2)[0]

        moving = ants.image_read(image)

        new = ants.apply_transforms(fixed, moving, transform)

        ants.image_write(new, f'{outDir}/{base}_registered.nii', ri=False)
