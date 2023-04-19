"""Register each run to one reference run."""

import os
import subprocess
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

    inDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/03_upsample'
    # Create output directory
    outDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/04_motionCorrect'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")

    NII_NAMES = sorted(glob.glob(f'{inDir}/*avg*'))

    # Create this file manually in ITK-SNAP (C chape centered at occipical)
    MASK_FILE = f"{inDir}/{sub}_ses-T2s_crop_regmask_ups2X.nii.gz"

    # =============================================================================
    print("Step_04: Register average echoes (motion correction).")

    for i in range(1, len(NII_NAMES)):
        # -------------------------------------------------------------------------
        # Compute affine transformation matrix
        # -------------------------------------------------------------------------
        # Prepare inputs
        in_fixed = NII_NAMES[0]  # Keep target image constant
        in_moving = NII_NAMES[i]
        in_mask = MASK_FILE

        # Prepare output
        basename, ext = in_moving.split(os.extsep, 1)
        basename = os.path.basename(basename)
        out_affine = os.path.join(outDir, "{}_affine.mat".format(basename))

        # Prepare command
        command1 = "greedy "
        command1 += "-d 3 "
        command1 += "-a -dof 6 "  # 6=rigid, 12=affine
        command1 += "-m NCC 2x2x2 "
        command1 += "-i {} {} ".format(in_fixed, in_moving)  # fixed moving
        command1 += "-o {} ".format(out_affine)
        command1 += "-ia-image-centers "
        command1 += "-n 100x50x10 "
        command1 += "-mm {} ".format(in_mask)
        command1 += "-float "

        # Execute command
        subprocess.run(command1, shell=True)

        # -------------------------------------------------------------------------
        # Apply affine transformation matrix
        # -------------------------------------------------------------------------
        # Prepare output
        basename, ext = in_moving.split(os.extsep, 1)
        basename = os.path.basename(basename)
        print(basename)
        out_moving = os.path.join(outDir, "{}_reg.nii.gz".format(basename))

        command2 = "greedy "
        command2 += "-d 3 "
        command2 += "-rf {} ".format(in_fixed)  # reference
        command2 += "-ri LINEAR "
        command2 += "-rm {} {} ".format(in_moving, out_moving)  # moving resliced
        command2 += "-r {} ".format(out_affine)

        print(command2)

        # Execute command
        subprocess.run(command2, shell=True)

print('\n\nFinished.')
