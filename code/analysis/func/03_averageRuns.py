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

SUBS = ['sub-08']
SESSIONS = ['ses-01']

for sub in SUBS:
    # Create subject-directory in derivatives if it does not exist
    subDir = f'{DATADIR}/derivatives/{sub}'

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
