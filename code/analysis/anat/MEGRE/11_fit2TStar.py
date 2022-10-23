"""Monoexponential decay (T2star) fitting."""

import os
import nibabel as nb
import numpy as np
# from scipy.linalg import lstsq
import glob
# =============================================================================
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
# Set subs to work on
SUBS = ['sub-05']

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
    # Create output directory
    outDir = f'{DATADIR}/derivatives/{sub}/{ses}/anat/megre/11_T2star'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Output directory is created")

    # Parameters
    NII_NAME = f"{inDir}/{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed.nii.gz"


    # =============================================================================
    print("Step_12: Fit T2*.")


    nii = nb.load(NII_NAME)
    dims = nii.shape
    nr_voxels = dims[0]*dims[1]*dims[2]

    data = nii.get_fdata()
    data = data.reshape((nr_voxels, dims[3]))

    # Take logarithm
    data = np.log(data, where=data > 0)

    design = np.ones((dims[3], 2))
    design[:, 0] *= -TEs

    betas = np.linalg.lstsq(design, data.T, rcond=None)[0]
    data = None

    np.max(betas[0])
    np.min(betas[0])

    np.max(betas[1])
    np.min(betas[1])

    T2_star = np.reciprocal(betas[0], where=betas[0] != 0)

    T2_star = np.abs(T2_star)
    T2_star[T2_star > 100] = 100  # Max clipping

    S0 = np.exp(betas[1])

    # Reshape to image space
    T2_star = T2_star.reshape((dims[0], dims[1], dims[2]))
    S0 = S0.reshape((dims[0], dims[1], dims[2]))

    # Save
    basename, ext = NII_NAME.split(os.extsep, 1)
    basename = os.path.basename(basename)
    img = nb.Nifti1Image(T2_star, affine=nii.affine)
    nb.save(img, os.path.join(outDir, "{}_T2s.nii.gz".format(basename)))
    img = nb.Nifti1Image(S0, affine=nii.affine)
    nb.save(img, os.path.join(outDir, "{}_S0.nii.gz".format(basename)))

print('Finished.')
