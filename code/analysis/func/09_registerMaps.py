'''

Registering statistical maps to anatomical data

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

    # outFolder = f'{DATADIR}/{sub}/ses-01/registration'

    # =========================================================================
    # Apply inverse transform
    # =========================================================================

    statMaps = sorted(glob.glob(f'{DATADIR}/{sub}/statMaps/*s.nii'))
    for statMap in statMaps:

        # Take care: fixed and moving are flipped
        fixed = glob.glob(f'{anatDir}/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz')[0]

        moving = statMap
    
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

        # =========================================================================
        # Crop map
        # =========================================================================

        inFile = f'{moving.split(".")[0]}_registered.nii'
        base = inFile.split('.')[0]
        outFile = f'{base}_crop.nii'

        command = 'fslroi '
        command += f'{inFile} '
        command += f'{outFile} '
        # command += '263 162 35 162 79 158'
        command += '271 162 7 162 31 159'

        subprocess.run(command,shell=True)
