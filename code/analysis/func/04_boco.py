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
sys.path.append('./code/misc')

from findTr import *

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
afniPath = '/Users/sebastiandresbach/abin'
layniiPath = '/Users/sebastiandresbach/git/laynii'

UPFACTOR = 4

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
afniPath = '/Users/sebastiandresbach/abin'
antsPath = '/Users/sebastiandresbach/ANTs/install/bin'

SUBS = ['sub-08']

SESSIONS = ['ses-01','ses-03','ses-04','ses-05']
SESSIONS = ['ses-01','ses-02']

for sub in SUBS:
    # Create subject-directory in derivatives if it does not exist
    subDir = f'{ROOT}/derivatives/{sub}'

    # Check for sessions with long ITI
    longITIses = 0
    allRuns = sorted(glob.glob(f'{subDir}/ses-*/func/{sub}_ses-*_task-stimulation_run-01_part-mag_cbv_moco.nii.gz'))
    for run in allRuns:
        nii = nb.load(run)
        nrTRs = nii.header['dim'][4]
        if nrTRs > 240:
            longITIses = longITIses+1

    if longITIses == 2:
        skipLongITI = True
    else:
        skipLongITI = False

    for ses in SESSIONS:
        print(f'Processing {ses}')

        outFolder = f'{ROOT}/derivatives/{sub}/{ses}/func'

        if skipLongITI:
            # Load first run to see whether it had a long ITI
            firstRun = f'{outFolder}/{sub}_{ses}_task-stimulation_run-01_part-mag_cbv_moco.nii.gz'
            nii = nb.load(firstRun)
            nrTRs = nii.header['dim'][4]
            if nrTRs > 240:
                print(f'Skipping sessions with long ITIs')
                continue


        tr = findTR(f'code/stimulation/{sub}/{ses}/{sub}_{ses}_run-01_neurovascularCoupling.log')
        tr = tr
        print(f'Effective TR: {tr} seconds')
        tr = tr/UPFACTOR
        print(f'Nominal TR will be: {tr} seconds')

        # for modality in ['bold']:
        for modality in ['bold', 'cbv']:

            # =====================================================================
            # Temporal upsampling
            # =====================================================================

            command = f'{afniPath}/3dUpsample '
            command += f'-overwrite '
            command += f'-datum short '
            command += f'-prefix {outFolder}/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz '
            command += f'-n {UPFACTOR} '
            command += f'-input {outFolder}/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}.nii'
            subprocess.call(command, shell=True)


            # fix TR in header
            subprocess.call(
                f'3drefit -TR {tr} '
                + f'{outFolder}'
                + f'/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz',
                shell=True
                )

            # =====================================================================
            # Duplicate first BOLD timepoint to match timing between cbv and bold
            # =====================================================================

            if modality == 'bold':
                nii = nb.load(
                    f'{outFolder}'
                    + f'/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz'
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
                    + f'/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz'
                    )

        # ==========================================================================
        # BOLD-correction
        # ==========================================================================

        cbvFile = f'{outFolder}/{sub}_{ses}_task-stimulation_run-avg_part-mag_cbv_intemp.nii.gz'
        boldFile = f'{outFolder}/{sub}_{ses}_task-stimulation_run-avg_part-mag_bold_intemp.nii.gz'

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
            + f'/{sub}_{ses}_task-stimulation_run-avg_part-mag_vaso_intemp.nii.gz'
            )

        # ==========================================================================
        # QA
        # ==========================================================================
        for modality in ['bold_intemp', 'vaso_intemp']:
            subprocess.run(
                f'{layniiPath}/LN_SKEW '
                + f'-input {outFolder}/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}.nii.gz',
                shell=True
                )

        # FSL has some hickups with values between 0 and 1. Therefore, we multiply
        # by 100.
        subprocess.run(
            f'fslmaths '
            + f'{outFolder}/{sub}_{ses}_task-stimulation_run-avg_part-mag_vaso_intemp.nii.gz '
            + f'-mul 100 '
            + f'{outFolder}/{sub}_{ses}_task-stimulation_run-avg_part-mag_vaso_intemp.nii.gz '
            + f'-odt short',
            shell=True
            )


    if skipLongITI:

        # for modality in ['bold']:
        for modality in ['bold', 'cbv']:

            # =====================================================================
            # Temporal upsampling
            # =====================================================================

            command = f'{afniPath}/3dUpsample '
            command += f'-overwrite '
            command += f'-datum short '
            command += f'-prefix {subDir}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz '
            command += f'-n {UPFACTOR} '
            command += f'-input {subDir}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}.nii'
            subprocess.call(command, shell=True)


            # fix TR in header
            subprocess.call(
                f'3drefit -TR {tr} '
                + f'{subDir}'
                + f'/{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz',
                shell=True
                )

            # =====================================================================
            # Duplicate first BOLD timepoint to match timing between cbv and bold
            # =====================================================================

            if modality == 'bold':
                nii = nb.load(
                    f'{subDir}'
                    + f'/{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz'
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
                nb.save(img, f'{subDir}'
                    + f'/{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz'
                    )

        # ==========================================================================
        # BOLD-correction
        # ==========================================================================

        cbvFile = f'{subDir}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_cbv_intemp.nii.gz'
        boldFile = f'{subDir}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_bold_intemp.nii.gz'

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
            img, f'{subDir}'
            + f'/{sub}_ses-avg_task-stimulation_run-avg_part-mag_vaso_intemp.nii.gz'
            )

        # ==========================================================================
        # QA
        # ==========================================================================
        for modality in ['bold_intemp', 'vaso_intemp']:
            subprocess.run(
                f'{layniiPath}/LN_SKEW '
                + f'-input {subDir}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}.nii.gz',
                shell=True
                )

        # FSL has some hickups with values between 0 and 1. Therefore, we multiply
        # by 100.
        subprocess.run(
            f'fslmaths '
            + f'{subDir}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_vaso_intemp.nii.gz '
            + f'-mul 100 '
            + f'{subDir}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_vaso_intemp.nii.gz '
            + f'-odt short',
            shell=True
            )
