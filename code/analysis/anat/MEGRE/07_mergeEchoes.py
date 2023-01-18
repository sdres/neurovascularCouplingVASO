

"""Split each echo to prepare for registration."""

import os
import subprocess
import numpy as np
import nibabel as nb
import glob
# =============================================================================
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# Set subs to work on
SUBS = ['sub-08']
DIRECTIONS = ['AP', 'PA', 'RL', 'LR']

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

    inDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/06_applyReg'
    # Create output directory
    outDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/07_mergeEchoes'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")


    for dir in DIRECTIONS:
        print(f'Processing {dir}')
        if dir == 'AP':
            NII_NAMES = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/05_upsampleEchoes/*{dir}*'))
        else:
            NII_NAMES = sorted(glob.glob(f'{inDir}/*{dir}*'))

        # =============================================================================
        print(f"Processing {dir}: Merge echoes.")

        # Average across echoes
        dims = nb.load(NII_NAMES[0]).shape

        temp = np.zeros(dims + (6,))

        for j in range(len(NII_NAMES)):
            print(f'Adding file {j+1}')
            # Load data
            nii = nb.load(NII_NAMES[j])
            temp[..., j] = np.squeeze(np.asanyarray(nii.dataobj))

            # Save echos as timeseries
            out_name = os.path.join(outDir, f'{sub}_{ses}_T2s_run-01_dir-{dir}_part-mag_MEGRE_crop_ups2X_prepped.nii.gz')

            img = nb.Nifti1Image(temp, affine=nii.affine, header=nii.header)
            nb.save(img, out_name)

print('  Finished.')
