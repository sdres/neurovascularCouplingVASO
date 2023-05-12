"""Detect voxels that does not decay over time."""

import os
import nibabel as nb
import numpy as np
import glob

# =============================================================================
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# Set subs to work on
SUBS = ['sub-07']

# =============================================================================
# Processing
for sub in SUBS:
    # Collecting files
    NII_NAMES = sorted(glob.glob(f'{DATADIR}/{sub}/*/anat/{sub}_ses-*_T2s_run-01_dir-*_echo-*_part-mag_MEGRE.nii.gz'))

    # Find MEGRE session of participant
    for i in range(1, 6):  # We had a maximum of 5 sessions
        if f'ses-0{i}' in NII_NAMES[0]:
            ses = f'ses-0{i}'

    inDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/09_composite'
    # Create output directory
    outDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/10_decayfix'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")

    # Parameters
    NII_NAME = f"{inDir}/{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite.nii.gz"

    # =============================================================================
    print("Step_10: Detect and fix non-decaying timepoints.")

    # =============================================================================

    nii = nb.load(NII_NAME)
    dims = nii.shape
    data = nii.get_fdata()

    data = np.abs(data)
    idx = data != 0
    data[idx] = np.log(data[idx])

    # 1-neighbour fix
    temp1 = np.zeros(dims[:-1])
    for i in range(dims[3] - 1):
        temp2 = data[..., i] - data[..., i+1]
        idx = temp2 < 0
        if (i > 0) and (i < dims[3] - 1):
            data[idx, i] = (data[idx, i-1] + data[idx, i+1]) / 2
        else:
            temp1[idx] = 1

    # Save
    basename, ext = NII_NAME.split(os.extsep, 1)
    basename = os.path.basename(basename)
    img = nb.Nifti1Image(temp1, affine=nii.affine)
    nb.save(img, os.path.join(outDir, "{}_decaymask.nii.gz".format(basename)))
    data = np.exp(data)
    img = nb.Nifti1Image(data, affine=nii.affine)
    nb.save(img, os.path.join(outDir, "{}_decayfixed.nii.gz".format(basename)))

print('Finished.')
