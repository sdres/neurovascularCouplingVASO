"""Average runs with same phase encoding axis."""

import os
import numpy as np
import nibabel as nb
import glob

# =============================================================================
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# Set subs to work on
SUBS = ['sub-06']
DIRECTIONS = [['AP', 'PA'], ['RL', 'LR']]
AXES = ['My', 'Mx']
# =============================================================================
# Processing
for sub in SUBS:
    # Collecting files
    NII_NAMES = sorted(glob.glob(f'{DATADIR}/{sub}/*/anat/{sub}_ses-*_T2s_run-01_dir-*_echo-*_part-mag_MEGRE.nii.gz'))

    # Find MEGRE session of participant
    for i in range(1,6):
        for i in range(1,6):  # We had a maximum of 5 sessions
            if f'ses-0{i}' in NII_NAMES[0]:
                ses = f'ses-0{i}'

    inDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/07_mergeEchoes'
    # Create output directory
    outDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/08_average'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")

    for axis, dirs in zip(AXES, DIRECTIONS):
        NII_NAMES = []

        for dir in dirs:
            NII_NAMES.append(f'{inDir}/{sub}_{ses}_T2s_run-01_dir-{dir}_part-mag_MEGRE_crop_ups2X_prepped.nii.gz')


        outName = f"{sub}_ses-T2s_dir-{axis}_part-mag_MEGRE_crop_ups2X_prepped_avg"

        # =============================================================================
        print(f"Average {axis}.")

        # =============================================================================

        # Load first file
        nii = nb.load(NII_NAMES[0])
        data = np.squeeze(nii.get_fdata())
        for i in range(1, len(NII_NAMES)):
            nii = nb.load(NII_NAMES[i])
            data += np.squeeze(nii.get_fdata())
        data /= len(NII_NAMES)

        # Save
        img = nb.Nifti1Image(data, affine=nii.affine)
        nb.save(img, os.path.join(outDir, "{}.nii.gz".format(outName)))

print('Finished.')
