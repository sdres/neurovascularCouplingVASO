"""Combine for flow artifact mitigated average (composite) image."""

import os
import numpy as np
import nibabel as nb
import glob

# =============================================================================
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# Set subs to work on
SUBS = ['sub-07']
DIRECTIONS = [['AP', 'PA'], ['RL', 'LR']]
AXES = ['My', 'Mx']

# =============================================================================
# Processing
for sub in SUBS:
    # Collecting files
    NII_NAMES = sorted(glob.glob(f'{DATADIR}/{sub}/*/anat/{sub}_ses-*_T2s_run-01_dir-*_echo-*_part-mag_MEGRE.nii.gz'))

    # Find MEGRE session of participant
    for i in range(1, 6):  # We had a maximum of 5 sessions
        if f'ses-0{i}' in NII_NAMES[0]:
            ses = f'ses-0{i}'

    inDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/08_average'
    # Create output directory
    outDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/09_composite'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")

    NII_NAMES = sorted(glob.glob(f'{inDir}/*.nii.gz'))
    outName = f'{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite'

    # =============================================================================
    print("Step_10: Composite.")

    # Output directory
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    print("  Output directory: {}".format(outDir))

    # Load data
    nii1 = nb.load(NII_NAMES[0])
    nii2 = nb.load(NII_NAMES[1])
    data1 = np.squeeze(nii1.get_fdata())
    data2 = np.squeeze(nii2.get_fdata())

    # -----------------------------------------------------------------------------
    # Compositing
    diff = data1 - data2

    idx_neg = diff < 0
    idx_pos = diff > 0

    data1[idx_pos] -= diff[idx_pos]
    data2[idx_neg] += diff[idx_neg]

    # Average
    data1 += data2
    data1 /= 2.
    # -----------------------------------------------------------------------------

    # Save
    out_name = nii1.get_filename().split(os.extsep, 1)[0]
    img = nb.Nifti1Image(data1, affine=nii1.affine)
    nb.save(img, os.path.join(outDir, "{}.nii.gz".format(outName)))

print('Finished.')
