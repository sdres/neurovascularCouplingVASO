"""Get first echo for vessel segmentation"""

import os
import nibabel as nb
import numpy as np
# from scipy.linalg import lstsq
import glob
# =============================================================================
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# Set subs to work on
SUBS = ['sub-06']

TEs = np.arange(1,7)*3.8

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

    inDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/10_decayfix'

    # Parameters
    NII_NAME = f"{inDir}/{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed.nii.gz"


    # =============================================================================

    nii = nb.load(NII_NAME)
    dims = nii.shape
    nr_voxels = dims[0]*dims[1]*dims[2]

    data = nii.get_fdata()

    for echo in range(2):
        print(f'Getting echo {echo}')
        tmp = data[...,echo]

        # Save
        basename, ext = NII_NAME.split(os.extsep, 1)
        basename = os.path.basename(basename)
        img = nb.Nifti1Image(tmp, affine=nii.affine)
        nb.save(img, os.path.join(inDir, "{}_echo{}.nii.gz".format(basename,echo)))


print('Finished.')
