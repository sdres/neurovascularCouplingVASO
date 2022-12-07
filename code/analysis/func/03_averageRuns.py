'''

Averages runs with the same stimulation protocol to boost SNR.
Runs with short ITIs are averaged within a session.
Runs with long ITIs are averaged across sessions.

'''

import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import sys

sys.path.append('./code/misc')

from computeT1w import *

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
afniPath = '/Users/sebastiandresbach/abin'
antsPath = '/Users/sebastiandresbach/ANTs/install/bin'

SUBS = ['sub-05']
SESSIONS = ['ses-01','ses-03','ses-04','ses-05']
# SESSIONS = ['ses-03']

for sub in SUBS:
    # Create subject-directory in derivatives if it does not exist
    subDir = f'{DATADIR}/derivatives/{sub}'

    # ==========================================================================
    # Average within sessions
    # ==========================================================================

    for ses in SESSIONS:
        # Create session-directory in derivatives if it does not exist
        sesDir = f'{subDir}/{ses}/func'

        # Look for individual runs within session
        runs = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stim*_run-0*_part-mag_*.nii.gz'))

        outFolder = f'{DATADIR}/derivatives/{sub}/{ses}/func'

        for modality in ['cbv', 'bold']:

            # find all runs of participant
            allRuns = sorted(glob.glob(f'{outFolder}/{sub}_{ses}_task-stimulation_run-0*_part-mag_{modality}_moco-reg.nii'))
            if ses == 'ses-01':
                firstRun = sorted(glob.glob(f'{outFolder}/{sub}_{ses}_task-stimulation_run-01_part-mag_{modality}_moco.nii.gz'))
                allRuns.insert(0, firstRun[0])

            nrRuns = len(allRuns)

            # find highest number of volumes
            highstVolNr = 0

            for run in allRuns:
                nii = nb.load(run)
                header = nii.header
                dataShape = header.get_data_shape()
                nrVolumes = dataShape[-1]
                if nrVolumes > highstVolNr:
                    highstVolNr = nrVolumes

            newShape = (
                dataShape[0],
                dataShape[1],
                dataShape[2],
                highstVolNr
                )
            newData = np.zeros(newShape)
            divisor = np.zeros(newShape)

            for run in allRuns:
                nii = nb.load(run)
                header = nii.header
                data = nii.get_fdata()
                nrVolumes = data.shape[-1]


                newData[:,:,:,:nrVolumes] += data
                divisor[:,:,:,:nrVolumes] += 1

            newData = newData/divisor

            nii = nb.load(allRuns[0])
            header = nii.header
            affine = nii.affine

            # save image
            img = nb.Nifti1Image(newData, header=header, affine=affine)
            nb.save(img, f'{outFolder}/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}.nii')


        modalities = glob.glob(f'{outFolder}/{sub}_{ses}_task-stimulation_run-avg_part-mag_*.nii')

        t1w = computeT1w(modalities[0], modalities[1])

        # Get header and affine
        header = nb.load(modalities[0]).header
        affine = nb.load(modalities[0]).affine

        # And save the image
        img = nb.Nifti1Image(t1w, header = header, affine = affine)
        nb.save(img, f'{outFolder}/{sub}_{ses}_task-stimulation_run-avg_part-mag_T1w.nii')


    # ==========================================================================
    # Average within and between sessions (long ITI)
    # ==========================================================================

    for modality in ['cbv', 'bold']:

        # Find runs with long ITI
        allRuns = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/ses-*/func/{sub}_ses-*_task-stim*_run-0*_part-mag_{modality}_moco-reg.nii*'))
        longRuns = []

        for run in allRuns:
            nii = nb.load(run)
            nrTRs = nii.header['dim'][4]
            # print(nrTRs)
            if nrTRs > 250:
                longRuns.append(run)

        if any('ses-01' in s for s in longRuns):
            firstRun = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/ses-01/func/{sub}_ses-01_task-stimulation_run-01_part-mag_{modality}_moco.nii.gz'))
            longRuns.insert(0, firstRun[0])

        nrRuns = len(longRuns)

        # find highest number of volumes
        highstVolNr = 0

        for run in longRuns:
            nii = nb.load(run)
            header = nii.header
            dataShape = header.get_data_shape()
            nrVolumes = dataShape[-1]
            if nrVolumes > highstVolNr:
                highstVolNr = nrVolumes

        newShape = (
            dataShape[0],
            dataShape[1],
            dataShape[2],
            highstVolNr
            )
        newData = np.zeros(newShape)
        divisor = np.zeros(newShape)

        for run in longRuns:
            nii = nb.load(run)
            header = nii.header
            data = nii.get_fdata()
            nrVolumes = data.shape[-1]


            newData[:,:,:,:nrVolumes] += data
            divisor[:,:,:,:nrVolumes] += 1

        newData = newData/divisor

        nii = nb.load(longRuns[0])
        header = nii.header
        affine = nii.affine

        # save image
        img = nb.Nifti1Image(newData, header=header, affine=affine)
        nb.save(img, f'{DATADIR}/derivatives/{sub}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}.nii')


    modalities = glob.glob(f'{DATADIR}/derivatives/{sub}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_*.nii')

    t1w = computeT1w(modalities[0], modalities[1])

    # Get header and affine
    header = nb.load(modalities[0]).header
    affine = nb.load(modalities[0]).affine

    # And save the image
    img = nb.Nifti1Image(t1w, header = header, affine = affine)
    nb.save(img, f'{DATADIR}/derivatives/{sub}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_T1w.nii')
