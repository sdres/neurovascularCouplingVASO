"""Saving statistical maps from fsl GLM"""

import subprocess
import glob
import nibabel as nb
import numpy as np
# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

subs = ['sub-07','sub-08','sub-09']

statMapsDict = {1: '1', 2: '2', 3: '4', 4: '12', 5: '24'}

for sub in subs:

    statFolder = f'{DATADIR}/{sub}/statMaps'
    for modality in ['vaso']:
        for i in range(1, 6):
            statMap = sorted(glob.glob(f'{statFolder}/*{modality}*.gfeat/cope{i}.feat/stats/zstat*'))[0]

            outName = f'{statFolder}/{sub}_{modality}_stim_{statMapsDict[i]}s.nii.gz'

            command = f'cp {statMap} {outName}'
            subprocess.run(command, shell=True)


#
# # Get dims of full nii
# file = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/sub-06_ses-avg_task-stimulation_run-avg_part-mag_T1w.nii'
# nii = nb.load(file)
# data = nii.get_fdata()
# dims = data.shape
# new = np.zeros(dims)
#
# file = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/statMaps/sub-06_vaso_stim_24s.nii.gz'
# nii2 = nb.load(file)
# data2 = nii2.get_fdata()
#
# new[:, :74, :] = data2
#
# img = nb.Nifti1Image(new, header=nii.header, affine=nii.affine)
# nb.save(img, '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/statMaps/sub-06_vaso_stim_24s_test.nii.gz')