'''

Registers and averages MP2RAGE inv-2 images from different sessions.

'''

import ants
import os
import glob
import subprocess

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

SUBS = ['sub-05']
UPFACT = 4

for sub in SUBS:

    # Find inv-2 images in all sessions
    images = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/ses-*/anat/upsample/*inv-2*.nii.gz'))

    # Set and load reference image in antsPy
    fixedFile = images[0]
    fixed = ants.image_read(fixedFile)

    # Load mask file in antsPy
    maskFile = f"{DATADIR}/derivatives/{sub}/anat/{sub}_regmask_ups{UPFACT}X.nii.gz"
    mask = ants.image_read(maskFile)

    # Loop over remaining images
    for image in images[1:]:
        base = os.path.basename(image).rsplit('.', 2)[0]
        print(f'Processing {base}')
        base, ext = image.split(os.extsep, 1)

        # Load moving image
        movingFile = image
        moving = ants.image_read(movingFile)

        # Register
        mytx = ants.registration(fixed=fixed,
                         moving=moving,
                         type_of_transform='Rigid',
                         mask = mask
                         )

        # Save image for quality control
        warped_moving = mytx['warpedmovout']
        ants.image_write(warped_moving, f'{base}_reg.nii.gz', ri=False)

        # Save transformation matrix for later application
        command = f'cp {mytx["fwdtransforms"][0]} {base}_transform.mat'
        subprocess.run(command, shell = True)
