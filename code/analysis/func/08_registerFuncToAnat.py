'''

Registering anatomical to functional image because that seemed to work better
than the other way round.

Then applying the inverse transformation to register func to anat.

'''

import glob
import os
import subprocess

subs = ['sub-07']

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'


for sub in subs:

    # Defining folders
    funcDir = f'{DATADIR}/{sub}/'  # Location of functional data
    anatDir = f'{DATADIR}/{sub}/ses-01/anat'  # Location of anatomical data

    regFolder = f'{anatDir}/registrationFiles'  # Folder where output will be saved
    # Make output folder if it does not exist already
    if not os.path.exists(regFolder):
        os.makedirs(regFolder)

    # =========================================================================
    # Registration
    # =========================================================================

    moving = glob.glob(f'{anatDir}/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz')[0]

    fixed = f'{funcDir}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_T1w.nii'

    # Set up ants command
    command = 'antsRegistration '
    command += f'--verbose 1 '
    command += f'--dimensionality 3 '
    command += f'--float 0 '
    command += f'--collapse-output-transforms 1 '
    command += f'--interpolation BSpline[5] '
    command += f'--output [{regFolder}/registered1_,{regFolder}/registered1_Warped.nii,1] '
    command += f'--use-histogram-matching 0 '
    command += f'--winsorize-image-intensities [0.005,0.995] '
    command += f'--initial-moving-transform {anatDir}/objective_matrix.txt '
    command += f'--transform SyN[0.1,3,0] '
    command += f'--metric CC[{fixed}, {moving},1,2] '
    command += f'--convergence [60x10,1e-6,10] '
    command += f'--shrink-factors 2x1 '
    command += f'--smoothing-sigmas 1x0vox '
    command += f'-x {anatDir}/{sub}_registrationMask.nii.gz'
    # Run command
    subprocess.run(command,shell=True)

    # Prepare command to apply transform and check quality
    command = 'antsApplyTransforms '
    command += f'--interpolation BSpline[5] '
    command += f'-d 3 -i {moving} '
    command += f'-r {fixed} '
    command += f'-t {regFolder}/registered1_1Warp.nii.gz '
    command += f'-t {regFolder}/registered1_0GenericAffine.mat '
    command += f'-o {moving.split(".")[0]}_registered.nii'
    # Run command
    subprocess.run(command,shell=True)

    # =========================================================================
    # Apply inverse transform
    # =========================================================================

    # Take care: fixed and moving are flipped
    fixed = glob.glob(f'{anatDir}/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz')[0]

    moving = f'{funcDir}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_T1w.nii'

    command = 'antsApplyTransforms '
    command += f'--interpolation BSpline[5] '
    command += f'-d 3 '
    command += f'-i {moving} '
    command += f'-r {fixed} '
    command += f'-t {regFolder}/registered1_1InverseWarp.nii.gz '
    # IMPORTANT: We take the inverse transform!!!
    command += f'-t [{regFolder}/registered1_0GenericAffine.mat, 1] '
    command += f'-o {moving.split(".")[0]}_registered.nii'

    subprocess.run(command,shell=True)
