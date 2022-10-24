'''

Apply transformation to MP2RAGE uni images.

'''

import ants
import os
import glob

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

SUBS = ['sub-05']

for sub in SUBS:

    # Find uni images in all sessions
    images = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/ses-*/anat/upsample/*uni*.nii.gz'))

    # Set and load reference image in antsPy
    fixedFile = images[0]
    fixed = ants.image_read(fixedFile)

    transforms = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/ses-*/anat/upsample/*transform.mat'))

    # =========================================================================
    # Apply transform

    for image, transform in zip(images[1:],transforms):
        base = os.path.basename(image).rsplit('.', 2)[0]
        print(f'Processing {base}')
        base, ext = image.split(os.extsep, 1)

        moving = ants.image_read(image)

        new = ants.apply_transforms(fixed, moving, transform)

        ants.image_write(new, f'{base}_registered.nii', ri=False)
