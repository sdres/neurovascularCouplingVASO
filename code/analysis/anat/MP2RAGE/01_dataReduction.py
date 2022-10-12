'''

Removing non-brain tissue from MP2RAGE images to reduce data load.

'''

import os
import glob
import subprocess

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

SUBS = ['sub-05']

for sub in subs:

    # =========================================================================
    # Brain extraction

    # Find inv-2 images in all sessions
    images = sorted(glob.glob(f'{DATADIR}/{sub}/ses-*/anat/*inv-2*.nii.gz'))

    # Initiate list for sessions
    sessions = []

    # Look through all images to find sessions IDs
    for image in images:
        for i in range(1,99):
            if f'ses-{str(i).zfill(2)}' in image:
                # Append session ID if found
                sessions.append(f'ses-{str(i).zfill(2)}')

    # Loop over sessions
    for ses in sessions:
        print(f'Working on session: {ses}')

        # Make output folder for each session
        sesOutDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat'
        if not os.path.exists(sesOutDir):
            os.makedirs(sesOutDir)
            print("Session output directory is created")

        # Find inv-2 images of session
        inv2Images = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/anat/*inv-2*.nii.gz'))

        for image in inv2Images:
            inv2Base = os.path.basename(image).rsplit('.', 2)[0]

            # Performing brain extraction
            command = 'bet '
            command += f'{image} '
            command += f'{sesOutDir}/{inv2Base}_brain '
            command += f'-m'

            subprocess.run(command, shell = True)

        # Find UNI images
        uniImages = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/anat/*uni*.nii.gz'))

        # Applying brain mask to UNI images
        for image in uniImages:
            uniBase = os.path.basename(image).rsplit('.', 2)[0]

            command = 'fslmaths '
            command += f'{image} '
            command += f'-mul '
            command += f'{sesOutDir}/{inv2Base}_brain_mask.nii.gz '
            command += f'{sesOutDir}/{uniBase}_brain'

            subprocess.run(command, shell = True)
