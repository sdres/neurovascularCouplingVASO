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
