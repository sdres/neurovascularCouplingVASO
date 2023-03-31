"""

Doing motion correction and regestering multiple runs across sessions.

"""

import ants
import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import sys
sys.path.append('./code/misc')
from computeT1w import *

# Set some paths
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# =============================================================================
# Do motion correction within runs
# =============================================================================

# Set subjects to work on
subs = ['sub-07']
# Set sessions to work on
sessions = ['ses-04']
# sessions = ['ses-03']


for sub in subs:
    # Create subject-directory in derivatives if it does not exist
    subDir = f'{DATADIR}/derivatives/{sub}'
    if not os.path.exists(subDir):
        os.makedirs(subDir)
        print("Subject directory is created")

    for ses in sessions:
        # Create session-directory in derivatives if it does not exist
        sesDir = f'{subDir}/{ses}/func'
        if not os.path.exists(sesDir):
            os.makedirs(sesDir)
            print("Session directory is created")

        # Look for individual runs within session
        runs = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stim*_run-0*_part-mag_*.nii.gz'))

        # Set folder for motion traces
        motionDir = f'{sesDir}/motionParameters'
        # Make folder to dump motion traces if it does not exist
        if not os.path.exists(motionDir):
            os.makedirs(motionDir)
            print("Motion directory is created")

        # Loop over individual runs
        for j, run in enumerate(runs, start=1):

            # Get a base name that we will use
            base = os.path.basename(run).rsplit('.', 2)[0]
            print(f'Processing run {base}')

            # Find modality of run
            modality = base.split('_')[-1]

            # Load data
            nii = nb.load(run)  # Load nifti
            header = nii.header  # Get header
            affine = nii.affine  # Get affine
            data = nii.get_fdata()  # Load data as array

            # Overwrite first 3 volumes with volumes 4,5,6
            new = np.concatenate((data[..., 3:6], data[..., 3:]), axis=3)
            img = nb.Nifti1Image(new, header=header, affine=affine)
            nb.save(img, f'{sesDir}/{base}.nii.gz')

            # Make automatic motion mask
            print('Generating mask')
            command = f'3dAutomask -prefix {sesDir}/{base}_moma.nii.gz -peels 3 -dilate 2 {sesDir}/{base}.nii.gz'
            subprocess.run(command, shell=True)

            # Make reference image
            reference = np.mean(data[..., 3:6], axis=-1)
            # And save it
            img = nb.Nifti1Image(reference, header=header, affine=affine)
            nb.save(img, f'{sesDir}/{base}_reference.nii.gz')

            # Load reference in antsPy style
            fixed = ants.image_read(f'{sesDir}/{base}_reference.nii.gz')

            # Get motion mask
            mask = ants.image_read(f'{sesDir}/{base}_moma.nii.gz')

            # Load timeseries data in antsPy style
            ts = ants.image_read(f'{sesDir}/{base}.nii.gz')

            # Perform motion correction
            print(f'Starting with MOCO...')
            corrected = ants.motion_correction(ts, fixed=fixed, mask=mask)
            ants.image_write(corrected['motion_corrected'], f'{sesDir}/{base}_moco.nii.gz')

            # =================================================================
            # Saving motion traces
            print(f'Saving motion traces...')

            # Set folder for motion traces
            runMotionDir = f'{motionDir}/{base}'
            # Make folder to dump motion traces if it does not exist
            if not os.path.exists(runMotionDir):
                os.makedirs(runMotionDir)
                print("Runwise motion directory is created")

            # Save transformation matrix for later
            for vol, matrix in enumerate(corrected['motion_parameters']):
                mat = matrix[0]
                os.system(f"cp {mat} {runMotionDir}/{base}_vol{vol:03d}.mat")

            # =========================================================================
            # Compute T1w image in EPI space within run
            # When bold and cbv parts are motion corrected
            if j % 2 == 0:
                print(f'Computing T1w image')
                tmp = int(j / 2)
                tmpRuns = sorted(glob.glob(f'{sesDir}/{sub}_{ses}_*_run-0{tmp}_*_moco.nii.gz'))

                t1w = computeT1w(tmpRuns[0], tmpRuns[1])

                # Get header and affine
                header = nb.load(tmpRuns[0]).header
                affine = nb.load(tmpRuns[0]).affine

                # And save the image
                img = nb.Nifti1Image(t1w, header=header, affine=affine)
                nb.save(img, f'{sesDir}/{sub}_{ses}_task-stimulation_run-0{tmp}_part-mag_T1w.nii')


# =============================================================================
# Register run-wise T1w images to first run of ses-01
# =============================================================================

# Set subjects to work on
# subs = ['sub-05']
# # Set sessions to work on
# sessions = ['ses-01', 'ses-02']
# sessions = ['ses-05']

for sub in subs:
    for ses in sessions:
        outFolder = f'{DATADIR}/derivatives/{sub}/{ses}/func'

        # look for individual runs
        runs = sorted(
            glob.glob(
                f'{DATADIR}/{sub}/{ses}/func'  # Folder
                + f'/{sub}_{ses}_task-stim*_run-0*_part-mag_*.nii.gz'  # files
            )
            )

        base = os.path.basename(runs[0]).rsplit('.', 2)[0].split('_')
        tmp = base[0]
        for subString in base[1:-1]:
            tmp = tmp + f'_{subString}'
        base = tmp

        # Divide by two because we have vaso and bold for each run
        nrRuns = int(len(runs)/2)

        # Set name of reference image
        refImage = (
            sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/ses-01/func/'
                             f'{sub}_ses-01_*_run-01_*_T1w.nii')))[0]

        # Get basename of reference image
        refBase = os.path.basename(refImage).rsplit('.', 2)[0]

        # Load reference image in antsPy style
        fixed = ants.image_read(refImage)

        # Define motion mask
        mask = ants.image_read(
            sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/ses-01/func/'
                             f'{sub}_ses-01_task-stim*_run-01_part-mag_cbv_moma.nii.gz'))[0]
        )

        # Because we want to register each run to the first run of the first
        # we want to exclude the first run of the first session
        if ses == 'ses-01':
            firstRun = 2
        if ses == 'ses-02':
            firstRun = 1

        for runNr, run in zip(range(1, nrRuns+1), runs[::2]):

            base = os.path.basename(run).rsplit('.', 2)[0].split('_')
            tmp = base[0]
            for subString in base[1:-1]:
                tmp = tmp + f'_{subString}'
            base = tmp

            # Define moving image
            moving = ants.image_read(
                f'{outFolder}'
                + f'/{base}_T1w.nii'
                )

            # Compute transformation matrix
            mytx = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform='Rigid',
                mask=mask
                )

            # Apply transformation
            mywarpedimage = ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=mytx['fwdtransforms'],
                interpolator='bSpline'
                )

            # Save image
            ants.image_write(
                mywarpedimage,
                f'{outFolder}'
                + f'/{base}_T1w_registered-{refBase}.nii')

            # Get transformation name
            transform1 = (
                f'{outFolder}'
                + f'/{base}_T1w_registered-{refBase}.mat'
                )

            # Save transform for future
            os.system(f"cp {mytx['fwdtransforms'][0]} {transform1}")

# =============================================================================
# Apply between run registration
# =============================================================================

# # Set subjects to work on
# subs = ['sub-05']
# # Set sessions to work on
# sessions = ['ses-01', 'ses-02']
# sessions = ['ses-04']

for sub in subs:
    for ses in sessions:
        outFolder = f'{DATADIR}/derivatives/{sub}/{ses}/func'

        for modality in ['cbv', 'bold']:

            # Look for individual runs
            runs = sorted(glob.glob(
                f'{DATADIR}/{sub}/{ses}/func'
                + f'/{sub}_{ses}_task-stim*_run-0*_part-mag_{modality}.nii.gz'
                )
                )

            # Find the number of runs within the session
            nrRuns = int(len(runs))

            # Set name of reference image
            refImage = (
                sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/ses-01/func/{sub}_ses-01_*_run-01_*_T1w.nii')))[0]

            # Get basename of reference image
            refBase = os.path.basename(refImage).rsplit('.', 2)[0].split('_')
            tmp = refBase[0]
            for subString in refBase[1:-1]:
                tmp = tmp + f'_{subString}'

            refBase = tmp

            # Get header and affine of reference image to easily assign later
            refHeader = nb.load(refImage).header
            refAffine = nb.load(refImage).affine

            # Load reference image in antspy style
            fixed = ants.image_read(refImage)

            # Load motion mask of reference image in antspy style
            mask = ants.image_read(
                f'{DATADIR}/derivatives/{sub}/ses-01/func'
                + f'/{refBase}_{modality}_moma.nii.gz'
                )

            # In the first session, we register to the first run, therefore
            # it would be omitted
            if ses == 'ses-01':
                firstRunId = 1
            else:
                firstRunId = 0

            # Loop over runs
            for run in runs[firstRunId:]:

                # Get the base name of the run
                runBase = os.path.basename(run).rsplit('.', 2)[0].split('_')
                tmp = runBase[0]
                for subString in runBase[1:-1]:
                    tmp = tmp + f'_{subString}'
                runBase = tmp

                # Load registration matrix between run and reference
                transformBetween = f'{outFolder}/{runBase}_T1w_registered-{refBase}_T1w.mat'

                # =============================================================
                # Separate run into individual volumes
                nii = nb.load(run)
                # get header and affine
                header = nii.header
                affine = nii.affine
                # Load data as array
                data = nii.get_fdata()

                # Loop over volumes
                for i in range(data.shape[-1]):
                    # Overwrite volumes 0,1,2 with volumes 3,4,5
                    if i <= 2:
                        vol = data[..., i+3]
                    else:
                        vol = data[..., i]

                    # Save individual volumes
                    img = nb.Nifti1Image(vol, header=header, affine=affine)
                    nb.save(img, f'{outFolder}/{runBase}_{modality}_vol{i:03d}.nii')

                # Loop over the volumes we just created to do the correction
                for i in range(data.shape[-1]):
                    # Load volume
                    moving = ants.image_read(f'{outFolder}/{runBase}_{modality}_vol{i:03d}.nii')

                    # Get within run transformation matrix of the volume
                    transformWithin = f'{outFolder}/motionParameters' \
                                      f'/{runBase}_{modality}/{runBase}_{modality}_vol{i:03d}.mat'

                    # Apply transformation matrices
                    mywarpedimage = ants.apply_transforms(
                        fixed=fixed,
                        moving=moving,
                        transformlist=[transformWithin, transformBetween],
                        interpolator='bSpline'
                        )

                    # Save warped image
                    ants.image_write(mywarpedimage, f'{outFolder}/{runBase}_{modality}_vol{i:03d}_warped.nii')

                # =============================================================
                # Assemble images of run
                newData = np.zeros(data.shape)
                for i in range(data.shape[-1]):
                    vol = nb.load(
                        f'{outFolder}'
                        + f'/{runBase}_{modality}_vol{i:03d}_warped.nii'
                    ).get_fdata()

                    newData[..., i] = vol

                img = nb.Nifti1Image(newData, header=refHeader, affine=refAffine)

                nb.save(img, f'{outFolder}/{runBase}_{modality}_moco-reg.nii')

                # Remove individual volumes
                os.system(f'rm {outFolder}/{runBase}_{modality}_vol*.nii')

# =============================================================================
# Get T1w image of registered runs
# =============================================================================

# Set subjects to work on
# subs = ['sub-05']
# sessions = ['ses-05']

for sub in subs:
    for ses in sessions:
        outFolder = f'{DATADIR}/derivatives/{sub}/{ses}/func'

        # Look for individual runs
        runs = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stim*_run-0*_part-mag_*.nii.gz'))

        nrRuns = int(len(runs)/2)

        if ses == 'ses-01':
            firstRun = 2
        else:
            firstRun = 1

        for runNr in range(firstRun, nrRuns+1):

            modalities = glob.glob(f'{outFolder}/{sub}_{ses}_task-stim*_run-0{runNr}_part-mag_*_moco-reg.nii')

            t1w = computeT1w(modalities[0], modalities[1])

            # Get header and affine
            header = nb.load(modalities[0]).header
            affine = nb.load(modalities[0]).affine

            # And save the image
            img = nb.Nifti1Image(t1w, header=header, affine=affine)
            nb.save(img, f'{outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_T1w_reg.nii')
