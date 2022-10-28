'''

Correcting bias field of MP2RAGE images.

'''

import os
import glob
import subprocess
import ants


DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

SUBS = ['sub-07']

for sub in SUBS:

    # =========================================================================
    # Brain extraction

    # Find inv-2 images in all sessions
    images = sorted(glob.glob(f'{DATADIR}/{sub}/ses-*/anat/*uni*.nii.gz'))

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

        for imageType in ['uni', 'inv-2']:
            # Find uni images of session
            images = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/anat/*{imageType}*.nii.gz'))

            for image in images:
                base = os.path.basename(image).rsplit('.', 2)[0]
                print(f'Processing {base}')

                command = 'N4BiasFieldCorrection '
                command += f'-d 3 '
                command += f'-i {image} '
                command += f'-o [{sesOutDir}/{base}_N4cor.nii.gz, {sesOutDir}/{base}_biasField.nii.gz]'

                subprocess.run(command, shell = True)
