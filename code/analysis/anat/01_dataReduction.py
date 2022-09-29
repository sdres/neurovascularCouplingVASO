'''

Registers and averages MP2RAGE images from different sessions to increase SNR.

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
    images = sorted(glob.glob(f'{DATADIR}/{sub}/ses-*/anat/*inv-2*.nii.gz'))

    sessions = []

    for image in images:
        for i in range(1,99):
            if f'ses-{str(i).zfill(2)}' in image:
                sessions.append(f'ses-{str(i).zfill(2)}')

    for ses in sessions:
        sesOutDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat'
        if not os.path.exists(sesOutDir):
            os.makedirs(sesOutDir)
            print("Subject directory is created")

        # Find inv-2 images
        inv2Images = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/anat/*inv-2*.nii.gz'))

        for image in inv2Images:
            inv2Base = os.path.basename(image).rsplit('.', 2)[0]

            command = 'bet '
            command += f'{image} '
            command += f'{sesOutDir}/{inv2Base}_brain '
            command += f'-m'

            subprocess.run(command, shell = True)

        # Find UNI images
        uniImages = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/anat/*uni*.nii.gz'))

        for image in uniImages:
            uniBase = os.path.basename(image).rsplit('.', 2)[0]

            command = 'fslmaths '
            command += f'{image} '
            command += f'-mul '
            command += f'{sesOutDir}/{inv2Base}_brain_mask.nii.gz '
            command += f'{sesOutDir}/{uniBase}_brain'

            subprocess.run(command, shell = True)
