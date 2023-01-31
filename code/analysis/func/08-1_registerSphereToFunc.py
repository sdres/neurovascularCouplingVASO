'''

Because the functional data in upsampled space is a burden to handle, we will
mask it with the sphere that we focused the segmentation on. For this, the sphere
has to be in functional space.

'''


import glob
import os
import subprocess

subs = ['sub-06']

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'


for sub in subs:

    # Defining folders
    funcDir = f'{DATADIR}/{sub}/ses-01/func'  # Location of functional data
    anatDir = f'{DATADIR}/{sub}/ses-01/anat'  # Location of anatomical data

    regFolder = f'{anatDir}/registrationFiles'  # Folder where output will be saved



    moving = f'{anatDir}/upsample/{sub}_LH_sphere_ups4X.nii.gz'
    fixed = f'{funcDir}/{sub}_ses-01_task-stimulation_run-avg_part-mag_T1w.nii'

    # Prepare command to apply transform and check quality
    command = 'antsApplyTransforms '
    command += f'--interpolation genericlabel '
    command += f'-d 3 -i {moving} '
    command += f'-r {fixed} '
    command += f'-t {regFolder}/registered1_1Warp.nii.gz '
    command += f'-t {regFolder}/registered1_0GenericAffine.mat '
    command += f'-o {moving.split(".")[0]}_registered.nii'
    # Run command
    subprocess.run(command,shell=True)


    # Mask T1w data as a test
    command = f'fslmaths {DATADIR}/{sub}/ses-01/func/{sub}_ses-01_task-stimulation_run-avg_part-mag_T1w.nii '
    command += f'-mul {DATADIR}/{sub}/ses-01/anat/upsample/{sub}_LH_sphere_ups4X_registered.nii '
    command += f'{DATADIR}/{sub}/ses-01/func/{sub}_ses-01_task-stimulation_run-avg_part-mag_T1w_masked-sphere'

    subprocess.run(command,shell=True)

    # =========================================================================
    # Apply inverse transform
    # =========================================================================

    # Take care: fixed and moving are flipped
    fixed = glob.glob(f'{anatDir}/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz')[0]

    moving = f'{funcDir}/{sub}_ses-01_task-stimulation_run-avg_part-mag_T1w_masked-sphere.nii.gz'

    command = 'antsApplyTransforms '
    command += f'--interpolation BSpline[5] '
    command += f'-d 3 '
    command += f'-i {moving} '
    command += f'-r {fixed} '
    command += f'-t {regFolder}/registered1_1InverseWarp.nii.gz '
    # IMPORTANT: We take the inverse transform!!!
    command += f'-t [{regFolder}/registered1_0GenericAffine.mat, 1] '
    command += f'-o {moving.split(".")[0]}_registered.nii.gz'

    subprocess.run(command,shell=True)

    command = f'fslmaths {moving.split(".")[0]}_registered.nii.gz -mul 1 {moving.split(".")[0]}_registered.nii.gz -odt float'
    subprocess.run(command,shell=True)


    # =========================================================================
    # Crop map
    inFile = f'{moving.split(".")[0]}_registered.nii.gz'
    base = inFile.split('.')[0]
    outFile = f'{base}_crop.nii.gz'

    command = 'fslroi '
    command += f'{inFile} '
    command += f'{outFile} '
    # command += '263 162 35 162 79 158' # sub-05
    # command += '271 162 7 162 31 159'  # Sub-06
    command =  '415 162 11 162 91 158'  # sub-09

    subprocess.run(command,shell=True)
