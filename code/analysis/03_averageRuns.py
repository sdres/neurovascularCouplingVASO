import os
import glob
import nibabel as nb
import numpy as np
import subprocess

root = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
afniPath = '/Users/sebastiandresbach/abin'
antsPath = '/Users/sebastiandresbach/ANTs/install/bin'


for sub in ['sub-03']:
    outFolder = f'{root}/derivatives/{sub}/'
    for acquiType in ['SingleShot', 'MultiShot']:
        for modality in ['cbv', 'bold']:

            # find all runs of participant
            allRuns = sorted(glob.glob(f'{root}/derivatives/{sub}/ses-0*/{sub}_ses-0*_task-stim{acquiType}_run-0*_part-mag_{modality}_moco_registered.nii'))
            firstRun = sorted(glob.glob(f'{root}/derivatives/{sub}/ses-01/{sub}_ses-01_task-stim{acquiType}_run-01_part-mag_{modality}_moco.nii'))
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
            nb.save(img, f'{outFolder}/{sub}_task-stim{acquiType}_part-mag_{modality}.nii')


        modalities = glob.glob(f'{outFolder}/{sub}_task-stim{acquiType}_part-mag_*.nii')
        print(modalities)
        # combining cbv and bold weighted images
        os.system(f'{afniPath}/3dTcat -prefix {outFolder}/{sub}_task-stim{acquiType}_part-mag_combined.nii  {modalities[0]} {modalities[1]} -overwrite')
        # Calculating T1w image in EPI space for each run
        os.system(f'{afniPath}/3dTstat -cvarinv -overwrite -prefix {outFolder}/{sub}_task-stim{acquiType}_part-mag_T1w.nii {outFolder}/{sub}_task-stim{acquiType}_part-mag_combined.nii')
        # Running biasfieldcorrection
        os.system(f'{antsPath}/N4BiasFieldCorrection -d 3 -i {outFolder}/{sub}_task-stim{acquiType}_part-mag_T1w.nii -o {outFolder}/{sub}_task-stim{acquiType}_part-mag_T1w_N4Corrected.nii')
