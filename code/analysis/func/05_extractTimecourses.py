''' Get event-related avergaes per stimulus duration '''

import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy import interpolate

import sys
# Define current dir
ROOT = os.getcwd()

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'
# Define subjects to work on
subs = ['sub-05']

# =============================================================================
# Extract timecourses
# =============================================================================

for sub in subs:
    for modality in ['vaso', 'bold']:

        run = f'{DATADIR}/{sub}/{sub}_task-stimulation_part-mag_{modality}_intemp.nii.gz'

        nii = nb.load(run)
        header = nii.header
        data = nii.get_fdata()

        mask = nb.load(f'{DATADIR}/{sub}/v1Mask.nii.gz').get_fdata()

        mask_mean = np.mean(data[:, :, :][mask.astype(bool)], axis=0)

        np.save(f'{DATADIR}/{sub}/{sub}_task-stimulation_part-mag_{modality}_intemp_timecourse',
                mask_mean
                )

# =============================================================================
# Upsample timecourses
# =============================================================================

# Define jitter
jitter = 6
# Divide by 2 because we already upsampled with a factor of 2 before
factor = jitter/2

for sub in subs:

    for modality in ['vaso', 'bold']:
        timecourse = np.load(f'{DATADIR}/{sub}/{sub}_task-stimulation_part-mag_{modality}_intemp_timecourse.npy')

        x = np.arange(0, timecourse.shape[0])

        for interpType in ['cubic', 'linear']:
            f = interpolate.interp1d(x, timecourse, fill_value='extrapolate', kind = interpType)

            xNew = np.arange(0, timecourse.shape[0], 1/factor)
            new = f(xNew)

            # Check whether interoplation makes sense
            # plt.figure()
            # plt.plot(x[:10], timecourse[:10], 'x')
            # plt.plot(xNew[:30], new[:30], 'o', alpha=0.5)
            # plt.show()

            # Save new timecourse
            np.save(f'{DATADIR}/{sub}/{sub}_task-stimulation_part-mag_{modality}_intemp_timecourse_intemp-{interpType}', new)
