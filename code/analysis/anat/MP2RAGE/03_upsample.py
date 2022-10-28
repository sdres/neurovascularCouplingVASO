'''

Upsample images. Assumes that a registration mask is generated based on first
session.

'''

import os
import subprocess
import glob

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
UPFACT = 4

SUBS = ['sub-07']

for sub in SUBS:

    # =============================================================================
    inDir = f"{DATADIR}/derivatives/{sub}/{ses}/anat"
    outDir = f"{DATADIR}/derivatives/{sub}/{ses}/anat/upsample"

    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")

    # Create this file manually in ITK-SNAP (C chape centered at occipical)
    # maskFile = f"{DATADIR}/derivatives/{sub}/anat/{sub}_regMask.nii.gz"

    # # -----------------------------------------------------------------------------
    # # Upsample registration mask
    # basename, ext = maskFile.split(os.extsep, 1)
    # basename = os.path.basename(basename)
    # out_file = os.path.join(outDir, f"{basename}_ups{UPFACT}X.nii.gz")
    #
    # # Prepare command
    # print("  Processing mask file...")
    # command1 = "c3d {} ".format(maskFile)
    # command1 += "-interpolation NearestNeighbor "
    # command1 += f"-resample {UPFACT}00% "
    # command1 += "-o {}".format(out_file)
    #
    # # Execute command
    # subprocess.run(command1, shell=True)

    # Find inv-2 images in all sessions
    images = sorted(glob.glob(f'{DATADIR}/{sub}/ses-*/anat/*inv-2*.nii.gz'))

    # Initiate list for sessions
    sessions = []

    # Look through all images to find sessions IDs
    for image in images:
        for i in range(1,99):
            if f'ses-{str(i).zfill(2)}' in image:
                # Append session ID if found
                sessions.append(f'ses-{str(i).zfill(2)}')

    # Loop over sessions
    for ses in sessions:
        print(f'Working on session: {ses}')

        # Make output folder for each session
        upsampleOutDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/upsample'
        if not os.path.exists(upsampleOutDir):
            os.makedirs(upsampleOutDir)
            print("Upsampling output directory is created")

        for imageType in ['uni', 'inv-2']:
            files = sorted(glob.glob(f'{DATADIR}/derivatives/{sub}/{ses}/anat/{sub}_*_{imageType}_*_N4cor_brain_crop.nii.gz'))

            # =============================================================================
            print(f'Upsample {imageType}.')

            for i, f in enumerate(files):
                # Prepare output
                basename, ext = f.split(os.extsep, 1)
                basename = os.path.basename(basename)
                out_file = os.path.join(upsampleOutDir, f"{basename}_ups{UPFACT}X.nii.gz")

                # Prepare command
                command2 = "c3d {} ".format(f)
                command2 += "-interpolation Cubic "
                command2 += f"-resample {UPFACT}00% "
                command2 += "-o {}".format(out_file)

                # Execute command
                subprocess.run(command2, shell=True)


    # # =============================================================================
    # # Upsample ROIs
    #
    # roiDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat'
    # images = sorted(glob.glob(f'{roiDir}/*sphere_crop.nii.gz'))
    #
    # for i, f in enumerate(images):
    #     # Prepare output
    #     basename, ext = f.split(os.extsep, 1)
    #     basename = os.path.basename(basename)
    #     out_file = os.path.join(roiDir, f"{basename}_ups{UPFACT}X.nii.gz")
    #
    #     # Prepare command
    #     command2 = "c3d {} ".format(f)
    #     command2 += "-interpolation NearestNeighbor "
    #     command2 += f"-resample {UPFACT}00% "
    #     command2 += "-o {}".format(out_file)
    #
    #     # Execute command
    #     subprocess.run(command2, shell=True)
    #
    # # =============================================================================
    # # Upsample segmentation
    #
    # roiDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat'
    # images = sorted(glob.glob(f'{roiDir}/*corrected_crop.nii.gz'))
    #
    # for i, f in enumerate(images):
    #     # Prepare output
    #     basename, ext = f.split(os.extsep, 1)
    #     basename = os.path.basename(basename)
    #     out_file = os.path.join(upsampleOutDir, f"{basename}_ups{UPFACT}X.nii.gz")
    #
    #     # Prepare command
    #     command2 = "c3d {} ".format(f)
    #     command2 += "-interpolation Multilabel "
    #     command2 += "-split -foreach "
    #     command2 += f"-resample {UPFACT}00% "
    #     command2 += "-smooth 0.5mm -endfor -merge "
    #     command2 += "-o {}".format(out_file)
    #
    #     # Execute command
    #     subprocess.run(command2, shell=True)
    #
    # print(f'\n\nFinished with {sub}.')
