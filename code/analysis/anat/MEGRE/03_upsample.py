"""Upsample images."""

import os
import subprocess
import numpy as np
import nibabel as nb
import glob


# =============================================================================
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# Set subs to work on
SUBS = ['sub-06']

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

    inDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/02_averageEchoes'
    # Create output directory
    outDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/03_upsample'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")

    NII_NAMES = sorted(glob.glob(f'{inDir}/*avg*'))

    # Create this file manually in ITK-SNAP (C chape centered at occipical)
    MASK_FILE = f"{DATADIR}/derivatives/{sub}/{ses}/anat/megre/{sub}_ses-T2s_crop_regmask.nii.gz"

    # =============================================================================
    print(f"Step_03: Upsample.")


    for i, f in enumerate(NII_NAMES):
        print("  Processing file {}...".format(i+1))
        # Prepare output
        basename, ext = f.split(os.extsep, 1)
        basename = os.path.basename(basename)
        out_file = os.path.join(outDir, "{}_ups2X.nii.gz".format(basename))

        # Prepare command
        command1 = "c3d {} ".format(f)
        command1 += "-interpolation Cubic "
        command1 += "-resample 200% "
        command1 += "-o {}".format(out_file)
        # Execute command
        subprocess.run(command1, shell=True)

    # -----------------------------------------------------------------------------
    # Also upsample registration mask
    basename, ext = MASK_FILE.split(os.extsep, 1)
    basename = os.path.basename(basename)
    out_file = os.path.join(outDir, "{}_ups2X.nii.gz".format(basename))

    # Prepare command
    print("  Processing mask file...")
    command2 = "c3d {} ".format(MASK_FILE)
    command2 += "-interpolation NearestNeighbor "
    command2 += "-resample 200% "
    command2 += "-o {}".format(out_file)
    # Execute command
    subprocess.run(command2, shell=True)

print('\n\nFinished.')
