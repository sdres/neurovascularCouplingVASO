'''

Registers MP2RAGE images from multiple sessions to increase SNR

'''


import ants
import os
import glob
import subprocess

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

subs = ['sub-03']

for sub in subs:

    outDir = f'{DATADIR}/derivatives/{sub}/anat'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Subject directory is created")



    # =========================================================================
    # Brain extraction

    # Find inv-2 images
    images = sorted(glob.glob(f'{DATADIR}/{sub}/ses-1*/anat/*inv-2*.nii.gz'))

    sessions = []

    for image in images:
        for i in range(1,99):
            if f'ses-{str(i).zfill(2)}' in image:
                sessions.append(f'ses-{str(i).zfill(2)}')

    for image in images:
        base = os.path.basename(image).rsplit('.', 2)[0]

        command = 'bet '
        command += f'{image} '
        command += f'{outDir}/{base}_brain '
        command += f'-m'

        subprocess.run(command, shell = True)

    # =========================================================================
    # Registration

    # Find brain extracted inv-2 images
    images = sorted(glob.glob(f'{outDir}/*inv-2*brain.nii.gz'))

    fixedFile = images[0]
    fixed = ants.image_read(fixedFile)

    maskFile = f'{outDir}/{sub}_ses-11_registrationMask.nii.gz'
    mask = ants.image_read(maskFile)

    for image in images[1:]:
        base = os.path.basename(image).rsplit('.', 2)[0]

        movingFile = image
        moving = ants.image_read(movingFile)

        mytx = ants.registration(fixed=fixed,
                         moving=moving,
                         type_of_transform='Rigid'
                         # mask = mask
                         )

        # Save image for quality control
        warped_moving = mytx['warpedmovout']
        ants.image_write(warped_moving, f'{outDir}/{base}_registered.nii', ri=False)

        # Save transformation
        command = f'cp {mytx["fwdtransforms"][0]} {outDir}/{base}_transform.mat'
        subprocess.run(command, shell = True)



    # =========================================================================
    # Apply transform

    for ses in sessions[1:]:
        images = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/anat/*.nii.gz'))
        for image in images:
            base = os.path.basename(image).rsplit('.', 2)[0]

            transform = glob.glob(f'{outDir}/*{ses}*_transform.mat')
            moving = ants.image_read(image)
            new = ants.apply_transforms(fixed, moving, transform)

            ants.image_write(new, f'{outDir}/{base}_registered.nii', ri=False)
