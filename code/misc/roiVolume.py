"""Calculate volume and surface area of perimeter chunk"""

import numpy as np
import nibabel as nb
import pandas as pd
import glob

SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']

subList = []
voxelCount = []
voxelVolume = []


for sub in SUBS:

    anatFolder = f'/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/{sub}/ses-01/anat/upsample'
    try:
        gmFile = f'{anatFolder}/{sub}_rim-LH_perimeter_chunk.nii.gz'
        test = nb.load(gmFile)
    except:
        gmFile = f'{anatFolder}/{sub}_rim-LH_perimeter_chunk.nii'
        test = nb.load(gmFile)

    base = gmFile.split('.')[0].split('/')[-1]

    peri = nb.load(gmFile)
    periData = peri.get_fdata()

    gm = (periData == 1).astype('int')
    edge = (periData == 2).astype('int')

    nrVox = np.sum(gm)
    volume = pow(0.175, 3) * nrVox
    muL = volume * 1000  # Volume in micro-litres
    surfaceArea = 3.141592653589793 * pow(10, 2)

    subList.append(sub)
    voxelCount.append(nrVox)
    voxelVolume.append(volume)
