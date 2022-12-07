"""Apply registration."""

import os
import subprocess
import numpy as np
import nibabel as nb
import glob

# =============================================================================
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# Set subs to work on
SUBS = ['sub-09']
DIRECTIONS = ['PA', 'RL', 'LR']

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

    inDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/05_upsampleEchoes'
    # Create output directory
    outDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/06_applyReg'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")


    REFERENCE = f"{DATADIR}/derivatives/{sub}/{ses}/anat/megre/03_upsample/{sub}_{ses}_T2s_run-01_dir-AP_echo-avg_part-mag_MEGRE_crop_ups2X.nii.gz"

    for dir in DIRECTIONS:
        NII_NAMES = sorted(glob.glob(f'{inDir}/*{dir}*'))
        AFFINE = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/04_motionCorrect/*{dir}*.mat'))[0]

        # =============================================================================
        print(f"Processing {dir}: Apply registration to each echo.")

        for i in range(0, len(NII_NAMES)):
            # -------------------------------------------------------------------------
            # Apply affine transformation matrix
            # -------------------------------------------------------------------------
            # Prepare inputs
            in_moving = NII_NAMES[i]
            affine = AFFINE

            # Prepare output
            basename, ext = in_moving.split(os.extsep, 1)
            basename = os.path.basename(basename)
            print(basename)
            out_moving = os.path.join(outDir, "{}_reg.nii.gz".format(basename))

            command = "greedy "
            command += "-d 3 "
            command += "-rf {} ".format(REFERENCE)  # reference
            command += "-ri LINEAR "  # No other better options than linear
            command += "-rm {} {} ".format(in_moving, out_moving)  # moving resliced
            command += "-r {} ".format(affine)

            # Execute command
            subprocess.run(command, shell=True)

print('\n\nFinished.')
