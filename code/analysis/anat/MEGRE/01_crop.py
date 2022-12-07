"""Reduce bounding box to decrease filesize."""

import os
import subprocess
import numpy as np
import nibabel as nb
import glob

# =============================================================================
# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# Set subs to work on
SUBS = ['sub-09']

# =============================================================================
# Processing

for sub in SUBS:
    # Collecting files
    NII_NAMES = sorted(glob.glob(f'{DATADIR}/{sub}/ses-*/anat/{sub}_ses-*_T2s_run-01_dir-*_echo-*_part-mag_MEGRE.nii.gz'))

    # Find MEGRE session of participant
    for i in range(1,6):
        for i in range(1,6):  # We had a maximum of 5 sessions
            if f'ses-0{i}' in NII_NAMES[0]:
                ses = f'ses-0{i}'

    # Create output directory
    outDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/01_crop'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")

    # =============================================================================
    # Check whether slices had to be ficed
    SLICEFIX = glob.glob(f'{DATADIR}/{sub}/{ses}/anat/*slicefix.nii*')
    if len(SLICEFIX) != 0:
        basename, ext = SLICEFIX[0].split(os.extsep, 1)
        search = basename[:-9]
        for i, name in enumerate(NII_NAMES):
            if search in name:
                NII_NAMES[i] = SLICEFIX[0]

    # # sub-05
    RANGE_X = [65, 365]  # xmin xsize
    RANGE_Y = [0, -1]  # ymin ysize
    RANGE_Z = [50, 150]  # zmin zsize

    # # sub-06
    RANGE_X = [60, 370]  # xmin xsize
    RANGE_Y = [0, -1]  # ymin ysize
    RANGE_Z = [55, 170]  # zmin zsize


    # # sub-08
    RANGE_X = [70, 360]  # xmin xsize
    RANGE_Y = [55, 145]  # ymin ysize
    RANGE_Z = [0, -1]  # zmin zsize

    # # sub-09
    RANGE_X = [55, 365]  # xmin xsize
    RANGE_Y = [0, -1]  # ymin ysize
    RANGE_Z = [45, 165]  # zmin zsize
    
    # =============================================================================
    for i, f in enumerate(NII_NAMES):
        print("  Processing file {} ...".format(i+1))
        # Prepare output
        basename, ext = f.split(os.extsep, 1)
        basename = os.path.basename(basename)
        if 'slicefix' in basename:
            basename = basename[:-9]

        out_file = os.path.join(outDir, "{}_crop.nii.gz".format(basename))

        # Prepare command
        command1 = "fslroi "
        command1 += "{} ".format(f)  # input
        command1 += "{} ".format(out_file)  # output
        command1 += "{} {} ".format(RANGE_X[0], RANGE_X[1])  # xmin xsize
        command1 += "{} {} ".format(RANGE_Y[0], RANGE_Y[1])  # ymin ysize
        command1 += "{} {} ".format(RANGE_Z[0], RANGE_Z[1])  # ymin ysize
        command1 += "0 -1 "  # tmin tsize
        # Execute command
        subprocess.run(command1, shell=True)

print('\n\nFinished.')
