"""Average echoes for before registration."""

import os
import numpy as np
import nibabel as nb
import glob

# =============================================================================
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
DIRECTIONS = ['AP', 'PA', 'RL', 'LR']
# Set subs to work on
SUBS = ['sub-09']

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

    inDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/01_crop'
    # Create output directory
    outDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/02_averageEchoes'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")

    for dir in DIRECTIONS:
        NII_NAMES = sorted(glob.glob(f'{inDir}/*{dir}*'))

        # =============================================================================
        print(f"{dir}: Average across echoes.")

        ECHOES = len(NII_NAMES)

        # Average across echoes
        for i, nii_name in enumerate(NII_NAMES):

            if i == 0:
                nii = nb.load(nii_name)
                header = nii.header
                affine = nii.affine

                data = nii.get_fdata()

            else:
                nii = nb.load(nii_name)
                tmp = nii.get_fdata()

                data += tmp

            data = data/ECHOES

            # save image
            img = nb.Nifti1Image(data, header=header, affine=affine)
            nb.save(img, f'{outDir}/{sub}_{ses}_T2s_run-01_dir-{dir}_echo-avg_part-mag_MEGRE_crop.nii.gz')


print('  Finished.')
