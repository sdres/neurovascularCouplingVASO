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
    # Upsample ROIs

    roiDir = f'{DATADIR}/derivatives/{sub}/segmentation'
    images = sorted(glob.glob(f'{roiDir}/*sphere.nii.gz'))

    outDir = f"{roiDir}/upsample"

    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")

    for i, f in enumerate(images):
        # Prepare output
        basename, ext = f.split(os.extsep, 1)
        basename = os.path.basename(basename)
        out_file = os.path.join(outDir, f"{basename}_ups{UPFACT}X.nii.gz")

        # Prepare command
        command2 = "c3d {} ".format(f)
        command2 += "-interpolation NearestNeighbor "
        command2 += f"-resample {UPFACT}00% "
        command2 += "-o {}".format(out_file)

        # Execute command
        subprocess.run(command2, shell=True)



    print(f'\n\nFinished with {sub}.')
