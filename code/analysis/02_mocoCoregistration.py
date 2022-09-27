'''

Doing motion correction and regestering multiple runs across sessions.

'''

import ants
import os
import glob
from nipype.interfaces import afni
import nibabel as nb
import numpy as np
import subprocess
from IPython.display import clear_output
import nipype.interfaces.fsl as fsl
import itertools
import pandas as pd

# Set some paths
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# ANTSPATH = '/Users/sebastiandresbach/ANTs/install/bin'

# =============================================================================
# Do motion correction within runs
# =============================================================================

# Set subjects to work on
subs = ['sub-03']
# Set sessions to work on
# sessions = ['ses-01', 'ses-02']
sessions = ['ses-01']

for sub in subs:
    # Create subject-directory in derivatives if it does not exist
    subDir = f'{DATADIR}/derivatives/{sub}'
    if not os.path.exists(subDir):
        os.makedirs(subDir)
        print("Subject directory is created")

    for ses in sessions:
        # Create session-directory in derivatives if it does not exist
        sesDir = f'{subDir}/{ses}'
        if not os.path.exists(sesDir):
            os.makedirs(sesDir)
            print("Session directory is created")

        # Look for individual runs within session
        runs = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stim*_run-0*_part-mag_*.nii.gz'))

        # Loop over individual runs
        for j, run in enumerate(runs, start = 1):
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


            # Make folder to save motion traces if it does not exist
            motionDir = f'{sesDir}/motionParameters/{base}'
            if not os.path.exists(motionDir):
                os.makedirs(motionDir)
                print("Motion directory is created")

            # Initiate lists for motion traces
            lstsubMot = []  # List for the motion value
            lstsubMot_Nme = []  # List for the motion name
            lstTR_sub = []  # List for name of subject.
            modalityList = []  # List for modality

            # Make moma
            print('Generating mask')
            subprocess.run(f'3dAutomask -prefix {sesDir}/{base}_moma.nii -peels 3 -dilate 2 {DATADIR}/{sub}/{ses}/func/{base}.nii.gz', shell=True)

            # Make reference image from volumes 5-7
            reference = np.mean(data[:,:,:,4:6], axis = -1)
            # And save it
            img = nb.Nifti1Image(reference, header = header, affine = affine)
            nb.save(img, f'{sesDir}/{base}_reference.nii')

            # Loop over and separate individual volumes
            for i in range(data.shape[-1]):
                # Overwrite volumes 0,1,2 with volumes 3,4,5
                # The first few volumes are not in steady state yet
                if i <= 2:
                    vol = data[:,:,:,i+3]
                else:
                    vol = data[:,:,:,i]

                # Save individual volume
                img = nb.Nifti1Image(vol, header=header, affine=affine)
                nb.save(img, f'{sesDir}/{base}_vol{i:03d}.nii')

            # Define motion-mask and reference images in 'antspy-style'
            mask = ants.image_read(f'{sesDir}/{base}_moma.nii')
            fixed = ants.image_read(f'{sesDir}/{base}_reference.nii')

            # Loop over volumes to do the correction
            for i in range(data.shape[-1]):
                # The current volume is the moving image
                moving = ants.image_read(f'{sesDir}/{base}_vol{i:03d}.nii')

                # Do the registration
                mytx = ants.registration(fixed = fixed,
                                         moving = moving,
                                         type_of_transform = 'Rigid',
                                         mask=mask
                                         )
                # Apply transformation
                mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving,transformlist=mytx['fwdtransforms'], interpolator='bSpline')
                # save warped image
                ants.image_write(mywarpedimage, f'{sesDir}/{base}_vol{i:03d}_warped.nii')

                # # ==============================================================
                # # Tracking motion traces
                # # ==============================================================
                #
                # # Save transformation matrix for later
                os.system(f"cp {mytx['fwdtransforms'][0]} {sesDir}/motionParameters/{base}/{base}_vol{i:03d}.mat")
                #
                # # Convert transformation matrix into FSL format
                # os.system(f'{ANTSPATH}/ConvertTransformFile 3 {sesDir}/motionParameters/{base}/{base}_vol{i:03d}.mat {sesDir}/motionParameters/{base}/{base}_vol{i:03d}_af.mat --convertToAffineType')
                # os.system(f'/usr/local/bin/c3d_affine_tool -ref {sesDir}/{base}_reference.nii -src {sesDir}/{base}_vol{i:03d}.nii -itk {sesDir}/motionParameters/{base}/{base}_vol{i:03d}_af.mat -ras2fsl -o {sesDir}/motionParameters/{base}/{base}_vol{i:03d}_FSL.mat -info-full')
                #
                # # Read motion parameters
                # tmp = fsl.AvScale(all_param=True,mat_file=f'{sesDir}/motionParameters/{base}/{base}_vol{i:03d}_FSL.mat');
                # tmpReadout = tmp.run();
                #
                # # Get the rotations (in rads) and translations (in mm) per volume
                # aryTmpMot = list(itertools.chain.from_iterable([tmpReadout.outputs.translations, tmpReadout.outputs.rot_angles]));
                #
                # # Save the rotation and translations in lists
                # lstsubMot.append(aryTmpMot)
                # lstTR_sub.append([int(i)+1 for k in range(6)])
                # lstsubMot_Nme.append([f'TX {modality}', f'TY {modality}', f'TZ {modality}', f'RX {modality}', f'RY {modality}', f'RZ {modality}'])
                # modalityList.append([modality for k in range(6)])
                #
                # clear_output(wait = True)

            # =================================================================
            # Assemble motion-correted images
            # =================================================================

            # Make array for new data
            newData = np.zeros(data.shape)

            # Loop over volumes again
            for i in range(data.shape[-1]):
                # Load current motion corrected volume
                vol = nb.load(f'{sesDir}/{base}_vol{i:03d}_warped.nii').get_fdata()
                # Assign volume to new data
                newData[:,:,:,i] = vol

            # Save data with original header and affine
            img = nb.Nifti1Image(newData, header = header, affine = affine)
            nb.save(img, f'{sesDir}/{base}_moco.nii')

            # Remove individual volumes
            os.system(f'rm {sesDir}/{base}_vol*.nii')

            # # ==============================================================
            # # Saving motion traces
            # # ==============================================================
            #
            # # Make appropriate arrays from lists
            # aryCurr = np.array(lstsubMot)
            # aryCurr_Ses =  aryCurr.reshape((aryCurr.size,-1))
            #
            # aryCurr_TR = np.array(lstTR_sub)
            # aryCurr_TR_Ses = aryCurr_TR.reshape((aryCurr_TR.size,-1))
            #
            # aryCurr_Nme = np.array(lstsubMot_Nme)
            # aryCurr_Nme_Ses = aryCurr_Nme.reshape((aryCurr_Nme.size,-1))
            #
            # aryIdx = np.arange(1,len(aryCurr_Nme_Ses)+1)
            #
            # aryCurr_mod = np.array(modalityList)
            # aryCurr_mod = aryCurr_mod.reshape((aryCurr_mod.size,-1))
            #
            # # Save into dictionary
            # data_dict = {
            #     'Time/TR': aryCurr_TR_Ses[:,0],
            #     'Motion_Name': aryCurr_Nme_Ses[:,0],
            #     'Motion': aryCurr_Ses[:,0],
            #     'idx':aryIdx,
            #     'modality': aryCurr_mod[:,0]}
            #
            # # Save motion parameters to disk as csv
            # pd_ses = pd.DataFrame(data=data_dict)
            # pd_ses.to_csv(f'{motionDir}/{base}_motionParameters.csv',
            #               index=False
            #               )

# =============================================================================
# Get run-wise T1w image in EPI space
# =============================================================================

# Set subjects to work on
subs = ['sub-03']
# Set sessions to work on
sessions = ['ses-01', 'ses-02']
sessions = ['ses-01']

for sub in subs:
    for ses in sessions:
        outFolder = f'{DATADIR}/derivatives/{sub}/{ses}'

        # Look for individual runs
        runs = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stim*_run-0*_part-mag_*.nii.gz'))

        # Divide by two because we have vaso and bold for each run
        nrRuns = int(len(runs)/2)

        for runNr, run in zip(range(1, nrRuns+1), runs[::2]):

            base = os.path.basename(run).rsplit('.', 2)[0].split('_')
            tmp = base[0]
            for subString in base[1:-1]:
                tmp = tmp + f'_{subString}'
            base = tmp

            print(f'processing {base}')
            # Get motion corrected data
            modalities = glob.glob(f'{outFolder}/{base}*_moco.nii')
            # Combining cbv and bold weighted images
            os.system(f'3dTcat -prefix {outFolder}/{base}_combined.nii  {modalities[0]} {modalities[1]} -overwrite')
            # Calculating T1w image in EPI space for each run
            os.system(f'3dTstat -cvarinv -overwrite -prefix {outFolder}/{base}_T1w.nii {outFolder}/{base}_combined.nii')
            # Running biasfieldcorrection
            os.system(f'N4BiasFieldCorrection -d 3 -i {outFolder}/{base}_T1w.nii -o {outFolder}/{base}_T1w_N4Corrected.nii')

# =============================================================================
# Register run-wise T1w images to first run of ses-01
# =============================================================================

# Set subjects to work on
subs = ['sub-03']
# Set sessions to work on
sessions = ['ses-01', 'ses-02']
sessions = ['ses-01']

for sub in subs:
    for ses in sessions:
        outFolder = f'{DATADIR}/derivatives/{sub}/{ses}'

        # look for individual runs
        runs = sorted(
            glob.glob(
            f'{DATADIR}/{sub}/{ses}/func' # folder
            + f'/{sub}_{ses}_task-stim*_run-0*_part-mag_*.nii.gz' # files
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
            f'{DATADIR}/derivatives/{sub}/ses-01/'
            + f'{base}_T1w_N4Corrected.nii'
            )

        # Get basename of reference image
        refBase = os.path.basename(refImage).rsplit('.', 2)[0]

        # Load reference image in antsPy style
        fixed = ants.image_read(refImage)

        # Define motion mask
        mask = ants.image_read(
            sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/ses-01/'
            + f'{sub}_ses-01_task-stim*_run-01_part-mag_cbv_moma.nii'))[0]
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
                + f'/{base}_T1w_N4Corrected.nii'
                )

            # Compute transofrmation matrix
            mytx = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform='Rigid',
                mask=mask
                )

            # Apply transofrmation
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
                + f'/{base}_T1w_N4Corrected_registered-{refBase}.nii')

            # Get transformation name
            transform1 = (
                f'{outFolder}'
                + f'/{base}_T1w_N4Corrected_registered-{refBase}.mat'
                )

            # Save transform for future
            os.system(f"cp {mytx['fwdtransforms'][0]} {transform1}")

# =============================================================================
# Apply between run registration
# =============================================================================

# Set subjects to work on
subs = ['sub-03']
# Set sessions to work on
sessions = ['ses-01', 'ses-02']
sessions = ['ses-01']

for sub in subs:
    for ses in sessions:
        outFolder = f'{DATADIR}/derivatives/{sub}/{ses}'
        for modality in ['cbv', 'bold']:

            # look for individual runs
            runs = sorted(glob.glob(
                f'{DATADIR}/{sub}/{ses}/func'
                + f'/{sub}_{ses}_task-stim*_run-0*_part-mag_{modality}.nii.gz'
                )
                )

            nrRuns = int(len(runs))

            refBase = os.path.basename(runs[0]).rsplit('.', 2)[0].split('_')
            tmp = refBase[0]
            for subString in refBase[1:-1]:
                tmp = tmp + f'_{subString}'

            refBase = tmp

            refImage = f'{DATADIR}/derivatives/{sub}/ses-01/{refBase}_T1w_N4Corrected.nii'
            refHeader = nb.load(refImage).header
            refAffine = nb.load(refImage).affine

            fixed = ants.image_read(refImage)

            mask = ants.image_read(
                f'{DATADIR}/derivatives/{sub}/ses-01'
                + f'/{refBase}_{modality}_moma.nii'
                )

            if ses == 'ses-01':
                firstRun = 2
            if ses == 'ses-02':
                firstRun = 1

            for run in runs[1:]:

                runBase = os.path.basename(run).rsplit('.', 2)[0].split('_')
                tmp = runBase[0]
                for subString in runBase[1:-1]:
                    tmp = tmp + f'_{subString}'
                runBase = tmp

                transform1 = f'{outFolder}/{runBase}_T1w_N4Corrected_registered-{refBase}_T1w_N4Corrected.mat'

                nii = nb.load(run)
                # get header and affine
                header = nii.header
                affine = nii.affine
                # Load data as array
                data = nii.get_fdata()

                # separate into individual volumes
                for i in range(data.shape[-1]):
                    # overwrite volumes 0,1,2 with volumes 3,4,5
                    if i <= 2:
                        vol = data[:,:,:,i+3]
                    else:
                        vol = data[:,:,:,i]

                    # Save individual volumes
                    img = nb.Nifti1Image(vol, header=header, affine=affine)
                    nb.save(img,
                    f'{outFolder}'
                    + f'/{runBase}_{modality}_vol{i:03d}.nii')

                # loop over volumes to do the correction
                for i in range(data.shape[-1]):
                    moving = ants.image_read(
                        f'{outFolder}'
                        + f'/{runBase}_{modality}_vol{i:03d}.nii')

                    # get within run transoformation matrix
                    transform2 = f'{outFolder}/motionParameters/{runBase}_{modality}/{runBase}_{modality}_vol{i:03d}.mat'

                    # apply transofrmation matrices
                    mywarpedimage = ants.apply_transforms(
                        fixed=fixed,
                        moving=moving,
                        transformlist = [transform2, transform1],
                        interpolator = 'bSpline'
                        )

                    # save warped image
                    ants.image_write(mywarpedimage,
                        f'{outFolder}'
                        + f'/{runBase}_{modality}_vol{i:03d}_warped.nii'
                        )

                # assemble images
                newData = np.zeros(data.shape)
                for i in range(data.shape[-1]):
                    vol = nb.load(
                    f'{outFolder}'
                    + f'/{runBase}_{modality}_vol{i:03d}_warped.nii'
                    ).get_fdata()

                    newData[:,:,:,i] = vol

                img = nb.Nifti1Image(newData,
                    header=refHeader,
                    affine=refAffine
                    )

                nb.save(
                    img,
                    f'{outFolder}'
                    + f'/{runBase}_{modality}_moco_registered.nii'
                    )

                # remove indvididual volumes
                os.system(f'rm {outFolder}/{runBase}_{modality}_vol*.nii')



# =============================================================================
# Get T1w image of registered runs
# =============================================================================

# Set subjects to work on
subs = ['sub-03']

for sub in subs:
    # for ses in ['ses-01', 'ses-02']:
    for ses in ['ses-01']:
        for acquiType in ['SingleShot', 'MultiShot']:
            # look for individual runs
            runs = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stim{acquiType}_run-0*_part-mag_*.nii.gz'))

            nrRuns = int(len(runs)/2)

            # if ses == 'ses-01':
            #     firstRun = 2
            # if ses == 'ses-02':
            #     firstRun = 1

            if acquiType == 'MultiShot':
                firstRun = 2
            if acquiType == 'SingleShot':
                firstRun = 1

            for runNr in range(firstRun, nrRuns+1):

                modalities = glob.glob(f'{DATADIR}/derivatives/{sub}/{ses}/{sub}_{ses}_task-stim{acquiType}_run-0{runNr}_part-mag_*_moco_registered.nii')

                os.system(f'3dTcat -prefix {outFolder}/{sub}_{ses}_task-stim{acquiType}_run-0{runNr}_part-mag_registered_combined.nii  {modalities[0]} {modalities[1]} -overwrite')
                # Calculating T1w image in EPI space for each run
                os.system(f'3dTstat -cvarinv -overwrite -prefix {outFolder}/{sub}_{ses}_task-stim{acquiType}_run-0{runNr}_part-mag_registered_T1w.nii {outFolder}/{sub}_{ses}_task-stim{acquiType}_run-0{runNr}_part-mag_registered_combined.nii')
                # Running biasfieldcorrection
                os.system(f'N4BiasFieldCorrection -d 3 -i {outFolder}/{sub}_{ses}_task-stim{acquiType}_run-0{runNr}_part-mag_registered_T1w.nii -o {outFolder}/{sub}_{ses}_task-stim{acquiType}_run-0{runNr}_part-mag_registered_T1w_N4Corrected.nii')
|
