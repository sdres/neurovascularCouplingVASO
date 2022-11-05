'''

Removing non-brain tissue from MP2RAGE images to reduce data load.

'''

import os
import glob
import subprocess

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

SUBS = ['sub-05']

BBOX = {'sub-05': {'ylower': 10, 'yrange': 110, 'zlower': 123, 'zrange': 177}}
BBOX = {'sub-06': {'ylower': 22, 'yrange': 110, 'zlower': 123, 'zrange': 177}}
BBOX = {'sub-07': {'ylower': 15, 'yrange': 115, 'zlower': 120, 'zrange': 180}}
BBOX = {'sub-08': {'ylower': 25, 'yrange': 125, 'zlower': 115, 'zrange': 155}}


for sub in SUBS:

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
        inv2Images = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/{ses}/anat/*inv-2*_N4cor.nii.gz'))

        for image in inv2Images:
            inv2Base = os.path.basename(image).rsplit('.', 2)[0]

            # Performing brain extraction
            command = 'bet '
            command += f'{image} '
            command += f'{sesOutDir}/{inv2Base}_brain '
            command += f'-m'

            subprocess.run(command, shell = True)

        # Find UNI images
        uniImages = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/{ses}/anat/*uni*_N4cor.nii.gz'))

        # Applying brain mask to UNI images
        for image in uniImages:
            uniBase = os.path.basename(image).rsplit('.', 2)[0]

            command = 'fslmaths '
            command += f'{image} '
            command += f'-mul '
            command += f'{sesOutDir}/{inv2Base}_brain_mask.nii.gz '
            command += f'{sesOutDir}/{uniBase}_brain'

            subprocess.run(command, shell = True)



        # =========================================================================
        # Cropping

        images = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/{ses}/anat/*_N4cor_brain.nii.gz'))

        tmpBBOX = BBOX[sub]

        for image in images:
            base = os.path.basename(image).rsplit('.', 2)[0]

            command = 'fslroi '
            command += f'{image} '
            command += f'{sesOutDir}/{base}_crop.nii.gz '
            command += f"0 207 {tmpBBOX['ylower']} {tmpBBOX['yrange']} {tmpBBOX['zlower']} {tmpBBOX['zrange']}"
            # break
            subprocess.run(command, shell = True)

        #
        # # =========================================================================
        # # Cropping segmentation and sphere if present
        #
        # images = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/{ses}/anat/{sub}_*_pveseg_corrected.nii.gz'))
        # images.append(f'{DATADIR}/derivatives/{sub}/{ses}/anat/{sub}_LH_sphere.nii.gz')
        #
        # tmpBBOX = BBOX[sub]
        #
        # for image in images:
        #     base = os.path.basename(image).rsplit('.', 2)[0]
        #
        #     command = 'fslroi '
        #     command += f'{image} '
        #     command += f'{sesOutDir}/{base}_crop.nii.gz '
        #     command += f"0 207 {tmpBBOX['ylower']} {tmpBBOX['yrange']} {tmpBBOX['zlower']} {tmpBBOX['zrange']}"
        #     # break
        #     subprocess.run(command, shell = True)
