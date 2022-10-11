"""Upsample images."""

import os
import subprocess
import numpy as np
import nibabel as nb
import glob

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
SUBS = ['sub-05']

for sub in SUBS:
    # =============================================================================
    OUTDIR = f"{ROOT}/derivatives/{sub}/anat/upsample"
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
        print("Output directory is created")
    # Create this file manually in ITK-SNAP (C chape centered at occipical)
    MASK_FILE = f"{ROOT}/derivatives/{sub}/anat/{sub}_regmask.nii.gz"


    # -----------------------------------------------------------------------------
    # Also upsample registration mask
    basename, ext = MASK_FILE.split(os.extsep, 1)
    basename = os.path.basename(basename)
    out_file = os.path.join(OUTDIR, "{}_ups2X.nii.gz".format(basename))

    # Prepare command
    print("  Processing mask file...")
    command2 = "c3d {} ".format(MASK_FILE)
    command2 += "-interpolation NearestNeighbor "
    command2 += "-resample 200% "
    command2 += "-o {}".format(out_file)
    # Execute command
    subprocess.run(command2, shell=True)

    for imageType in ['inv-2']:
        files = sorted(glob.glob(f'{ROOT}/derivatives/{sub}/*/anat/*{imageType}*brain.nii.gz'))


        # =============================================================================
        print(f"MP2RAGE Step 02: Upsample {imageType}.")

        # Output directory
        if not os.path.exists(OUTDIR):
            os.makedirs(OUTDIR)
        print("  Output directory: {}\n".format(OUTDIR))

        for i, f in enumerate(files):
            print("  Processing file {}...".format(i+1))
            # Prepare output
            basename, ext = f.split(os.extsep, 1)
            basename = os.path.basename(basename)
            out_file = os.path.join(OUTDIR, "{}_ups2X.nii.gz".format(basename))

            # Prepare command
            command1 = "c3d {} ".format(f)
            command1 += "-interpolation Cubic "
            command1 += "-resample 200% "
            command1 += "-o {}".format(out_file)
            # Execute command
            subprocess.run(command1, shell=True)


    print('\n\nFinished.')
