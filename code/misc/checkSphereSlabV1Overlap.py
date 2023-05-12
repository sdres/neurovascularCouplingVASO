"""Check if there is any part of V1 outside of the sphere but inside of the funcitonal coverage"""

import nibabel as nb
import numpy as np
import pandas as pd

folder = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

subList = []
coveredList = []
inSphereList = []

subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
subs = ['sub-08']

for sub in subs:
    sphereFile = f'{folder}/{sub}/ses-01/anat/upsample/{sub}_LH_sphere_ups4X.nii.gz'
    v1File = f'{folder}/rawdata_bv/{sub}/ses-01/anat/v1_registered-{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
    coverageFile = f'{folder}/{sub}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_T1w_registered.nii'

    # Get affine and header to save data later
    affine = nb.load(sphereFile).affine
    header = nb.load(sphereFile).header

    # Load data in numpy arrays
    sphereData = nb.load(sphereFile).get_fdata().astype(np.int_)
    v1Data = nb.load(v1File).get_fdata().astype(np.int_)
    coverageData = nb.load(coverageFile).get_fdata()

# sphereData.shape
# v1Data.shape
# coverageData.shape

    # Binarize coverage
    coverageData = (coverageData > 0).astype(np.int_)

    # Intersect v1 and coverage
    v1Coverage = coverageData & v1Data

    # Get percent of covered V1
    v1All = np.sum(v1Data)  # Nr. voxels of v1 mask
    v1Covered = np.sum(v1Coverage)  # Nr. voxels in functional slab
    v1CoveredPercent = (v1Covered/v1All) * 100
    print(f'We covered {v1CoveredPercent:.2f}% of V1 in our functional slab')

    # Get Part of V1 in the sphere
    sphereMinCovered = sphereData + v1Data
    inSphere = (sphereMinCovered == 2).astype(int)
    voxInSphere = np.sum(inSphere)
    sphereCoveredPercent = (voxInSphere/v1Covered) * 100
    print(f'Of this, {sphereCoveredPercent:.2f}% is in the sphere')

    # Get percentage of v1 in sphere
    percentInSphere = (voxInSphere/v1All) * 100
    print(f'So, {percentInSphere:.2f}% of v1 is in the sphere')

    subList.append(sub)
    coveredList.append(v1CoveredPercent)
    inSphereList.append(percentInSphere)

    # Get voxels outside of sphere
    sphereMinCovered = sphereData - v1Coverage
    notSphere = (sphereMinCovered == -1).astype(int)

    # =====================================================================================
    # Save masks

    # V1 inside of sphere
    newImg = nb.Nifti1Image(inSphere, header=header, affine=affine)
    outFile = f'{folder}/{sub}/v1InSphere.nii.gz'
    nb.save(newImg, outFile)

    # V1 not in sphere
    newImg = nb.Nifti1Image(notSphere, header=header, affine=affine)
    outFile = f'{folder}/{sub}/v1NotInSphere.nii.gz'
    nb.save(newImg, outFile)

data = pd.DataFrame({'subject': subList, 'v1Covered': coveredList, 'inSphere': inSphereList})