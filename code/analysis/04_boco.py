import subprocess
import glob
import os
import nibabel as nb
import numpy as np
import re

root = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
antsPath = '/Users/sebastiandresbach/ANTs/install/bin'
afniPath = '/Users/sebastiandresbach/abin'
layniiPath = '/Users/sebastiandresbach/git/laynii'

def findTR(logfile):
    with open(logfile) as f:
        f = f.readlines()

    triggerTimes = []
    for line in f[1:]:
        if re.findall("Keypress: 5",line):
            triggerTimes.append(float(re.findall("\d+\.\d+", line)[0]))

    triggerTimes[0] = 0

    triggersSubtracted = []
    for n in range(len(triggerTimes)-1):
        triggersSubtracted.append(float(triggerTimes[n+1])-float(triggerTimes[n]))

    meanFirstTriggerDur = np.mean(triggersSubtracted[::2])
    meanSecondTriggerDur = np.mean(triggersSubtracted[1::2])

    # find mean trigger-time
    meanTriggerDur = (meanFirstTriggerDur+meanSecondTriggerDur)/2
    return meanTriggerDur

# boco on mean subject mean
for sub in ['sub-01']:
    outFolder = f'{root}/derivatives/{sub}'

    tr = findTR(f'../stimulation/{sub}/ses-01/{sub}_ses-01_run-01_neurovascularCoupling.log')

    for modality in ['bold', 'cbv']:
        # temporal upsampling
        command = f'{afniPath}/3dUpsample '
        command += f'-overwrite '
        command += f'-datum short '
        command += f'-prefix {outFolder}/{sub}_task-stimulation_part-mag_{modality}_intemp.nii.gz '
        command += f'-n 2 '
        command += f'-input {outFolder}/{sub}_task-stimulation_part-mag_{modality}.nii'
        subprocess.call(command, shell=True)


        # fix TR in header
        subprocess.call(
            f'3drefit -TR {tr} '
            + f'{outFolder}'
            + f'/{sub}_task-stimulation_part-mag_{modality}_intemp.nii.gz',
            shell=True
            )

        if modality == 'bold':
            nii = nb.load(
                f'{outFolder}'
                + f'/{sub}_task-stimulation_part-mag_{modality}_intemp.nii.gz'
                )
            data = nii.get_fdata()
            header = nii.header
            affine = nii.affine

            newData = np.zeros(data.shape)

            for i in range(data.shape[-1]-1):
                if i == 0:
                    newData[:,:,:,i]=data[:,:,:,i]
                else:
                    newData[:,:,:,i]=data[:,:,:,i-1]
            img = nb.Nifti1Image(newData.astype(int), header=header, affine=affine)
            nb.save(img, f'{outFolder}'
                + f'/{sub}_task-stimulation_part-mag_{modality}_intemp.nii.gz'
                )


    # BOCO
    nii1 = nb.load(f'{outFolder}/{sub}_task-stimulation_part-mag_cbv_intemp.nii.gz').get_fdata()
    nii2 = nb.load(f'{outFolder}/{sub}_task-stimulation_part-mag_bold_intemp.nii.gz').get_fdata()
    header = nb.load(f'{outFolder}/{sub}_task-stimulation_part-mag_cbv_intemp.nii.gz').header
    affine = nb.load(f'{outFolder}/{sub}_task-stimulation_part-mag_cbv_intemp.nii.gz').affine
    new = np.divide(nii1[:,:,:,:-1], nii2[:,:,:,:-1])

    new[new > 1.5] = 1.5
    new[new < -1.5] = -1.5

    img = nb.Nifti1Image(new, header=header, affine=affine)
    nb.save(
        img, f'{outFolder}'
        + f'/{sub}_task-stimulation_part-mag_vaso_intemp.nii.gz'
        )

    # calculate quality measures
    for modality in ['bold_intemp', 'vaso_intemp']:
        subprocess.run(
            f'{layniiPath}/LN_SKEW '
            + f'-input {outFolder}/{sub}_task-stimulation_part-mag_{modality}.nii.gz',
            shell=True
            )

    subprocess.run(
        f'fslmaths '
        + f'{outFolder}/{sub}_task-stimulation_part-mag_vaso_intemp.nii.gz '
        + f'-mul 100 '
        + f'{outFolder}/{sub}_task-stimulation_part-mag_vaso_intemp.nii.gz '
        + f'-odt short',
        shell=True
        )
