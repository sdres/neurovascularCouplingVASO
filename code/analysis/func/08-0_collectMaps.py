"""Saving statistical maps from fsl GLM"""

import subprocess
import glob
import os

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
subs = ['sub-09']

statMapsDict = {1: '1', 2: '2', 3: '4', 4: '12', 5: '24'}

for sub in subs:
    statFolder = f'{DATADIR}/{sub}/statMaps'
    outFolder = f'{statFolder}/glm_fsl'

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
        print("Output directory is created")

    for modality in ['bold', 'vaso']:
        for i in range(1, 6):
            statMap = sorted(glob.glob(f'{statFolder}/*{modality}*.gfeat/cope{i}.feat/stats/cope*'))[0]

            outName = f'{outFolder}/{sub}_{modality}_stim_{statMapsDict[i]}s_cope.nii.gz'

            command = f'cp {statMap} {outName}'
            subprocess.run(command, shell=True)
