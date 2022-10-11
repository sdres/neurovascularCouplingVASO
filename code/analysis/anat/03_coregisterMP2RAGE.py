'''

Registers and averages MP2RAGE images from different sessions to increase SNR.

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

    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Subject directory is created")

    # =========================================================================
    # Registration

    # Find brain extracted inv-2 images
    images = sorted(glob.glob(f'{inDir}/*inv-2*brain_up*.nii.gz'))

    fixedFile = images[0]
    fixed = ants.image_read(fixedFile)

    # Create this file manually in ITK-SNAP (C chape centered at occipical)
    maskFile = f"{inDir}/{sub}_regmask_ups2X.nii.gz"

    mask = ants.image_read(maskFile)

    for image in images[1:]:
        base = os.path.basename(image).rsplit('.', 2)[0]
        # basename, ext = image.split(os.extsep, 1)

        movingFile = image
        moving = ants.image_read(movingFile)

        mytx = ants.registration(fixed=fixed,
                         moving=moving,
                         type_of_transform='Rigid',
                         mask = mask
                         )

        # Save image for quality control
        warped_moving = mytx['warpedmovout']
        ants.image_write(warped_moving, f'{outDir}/{base}_reg.nii.gz', ri=False)

        # Save transformation
        command = f'cp {mytx["fwdtransforms"][0]} {outDir}/{base}_transform.mat'
        subprocess.run(command, shell = True)


    # # =========================================================================
    # # Apply transform
    #
    # for ses in sessions[1:]:
    #     images = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/anat/*.nii.gz'))
    #     for image in images:
    #         base = os.path.basename(image).rsplit('.', 2)[0]
    #
    #         transform = glob.glob(f'{outDir}/*{ses}*_transform.mat')
    #         moving = ants.image_read(image)
    #         new = ants.apply_transforms(fixed, moving, transform)
    #
    #         ants.image_write(new, f'{outDir}/{base}_registered.nii', ri=False)
