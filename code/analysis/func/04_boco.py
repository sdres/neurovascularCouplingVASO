'''

Standard VASO-pipeline things like:
- temporal upsampling
- BOLD-correction
- QA

'''

import subprocess
import glob
import os
import nibabel as nb
import numpy as np
import re
import sys

# Define current dir
ROOT = os.getcwd()
sys.path.append(os.path.abspath('./code/analysis'))
from findTr import *

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
afniPath = '/Users/sebastiandresbach/abin'
layniiPath = '/Users/sebastiandresbach/git/laynii'

# boco on mean subject mean
for sub in ['sub-03']:
    outFolder = f'{ROOT}/derivatives/{sub}'
    for acquiType in ['SingleShot', 'MultiShot']:

        if acquiType == 'SingleShot':
            tr = findTR(f'code/stimulation/{sub}/ses-01/{sub}ses-01blockStim_30sOnOffrun-01.log')
        if acquiType == 'MultiShot':
            tr = findTR(f'code/stimulation/{sub}/ses-01/{sub}ses-01blockStim_30sOnOffrun-03.log')
        print(f'{acquiType} tr: {tr}')

        # for modality in ['bold']:
        for modality in ['bold', 'cbv']:

            # =====================================================================
            # Temporal upsampling
            # =====================================================================

            command = f'{afniPath}/3dUpsample '
            command += f'-overwrite '
            command += f'-datum short '
            command += f'-prefix {outFolder}/{sub}_task-stim{acquiType}_part-mag_{modality}_intemp.nii.gz '
            command += f'-n 2 '
            command += f'-input {outFolder}/{sub}_task-stim{acquiType}_part-mag_{modality}.nii'
            subprocess.call(command, shell=True)


            # fix TR in header
            subprocess.call(
                f'3drefit -TR {tr} '
                + f'{outFolder}'
                + f'/{sub}_task-stim{acquiType}_part-mag_{modality}_intemp.nii.gz',
                shell=True
                )

            # =====================================================================
            # Duplicate first BOLD timepoint to match timing between cbv and bold
            # =====================================================================

            if modality == 'bold':
                nii = nb.load(
                    f'{outFolder}'
                    + f'/{sub}_task-stim{acquiType}_part-mag_{modality}_intemp.nii.gz'
                    )

                # Load data
                data = nii.get_fdata()  # Get data
                header = nii.header  # Get header
                affine = nii.affine  # Get affine

                # Make new array
                newData = np.zeros(data.shape)

                for i in range(data.shape[-1]-1):
                    if i == 0:
                        newData[:,:,:,i]=data[:,:,:,i]
                    else:
                        newData[:,:,:,i]=data[:,:,:,i-1]

                # Save data
                img = nb.Nifti1Image(newData.astype(int), header=header, affine=affine)
                nb.save(img, f'{outFolder}'
                    + f'/{sub}_task-stim{acquiType}_part-mag_{modality}_intemp_test.nii.gz'
                    )

        # ==========================================================================
        # BOLD-correction
        # ==========================================================================

        cbvFile = f'{outFolder}/{sub}_task-stim{acquiType}_part-mag_cbv_intemp.nii.gz'
        boldFile = f'{outFolder}/{sub}_task-stim{acquiType}_part-mag_bold_intemp.nii.gz'

        # Load data
        nii1 = nb.load(cbvFile).get_fdata()  # Load cbv data
        nii2 = nb.load(boldFile).get_fdata()  # Load BOLD data
        header = nb.load(cbvFile).header  # Get header
        affine = nb.load(cbvFile).affine  # Get affine

        # Divide VASO by BOLD for actual BOCO
        new = np.divide(nii1[:,:,:,:-1], nii2[:,:,:,:-1])

        # Clip range to -1.5 and 1.5. Values should be between 0 and 1 anyway.
        new[new > 1.5] = 1.5
        new[new < -1.5] = -1.5

        # Save bold-corrected VASO image
        img = nb.Nifti1Image(new, header=header, affine=affine)
        nb.save(
            img, f'{outFolder}'
            + f'/{sub}_task-stim{acquiType}_part-mag_vaso_intemp.nii.gz'
            )

        # ==========================================================================
        # QA
        # ==========================================================================
        for modality in ['bold_intemp', 'vaso_intemp']:
            subprocess.run(
                f'{layniiPath}/LN_SKEW '
                + f'-input {outFolder}/{sub}_task-stim{acquiType}_part-mag_{modality}.nii.gz',
                shell=True
                )

        # FSL has some hickups with values between 0 and 1. Therefore, we multiply
        # by 100.
        subprocess.run(
            f'fslmaths '
            + f'{outFolder}/{sub}_task-stim{acquiType}_part-mag_vaso_intemp.nii.gz '
            + f'-mul 100 '
            + f'{outFolder}/{sub}_task-stim{acquiType}_part-mag_vaso_intemp.nii.gz '
            + f'-odt short',
            shell=True
            )
