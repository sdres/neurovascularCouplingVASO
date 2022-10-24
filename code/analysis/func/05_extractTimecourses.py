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
sys.path.append('./code/misc')
from findTr import *


ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'
# Define subjects to work on
subs = ['sub-05']
# sessions = ['ses-01', 'ses-03']
sessions = ['ses-01']


# Get TR
UPFACTOR = 4


# =============================================================================
# Extract mean timecourses
# =============================================================================

for sub in subs:
    for ses in sessions:

        for modality in ['vaso', 'bold']:
        # for modality in ['bold']:

            run = f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz'
            # run = f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}.nii'

            nii = nb.load(run)
            header = nii.header
            data = nii.get_fdata()

            mask = nb.load(f'{DATADIR}/{sub}/v1Mask.nii.gz').get_fdata()

            mask_mean = np.mean(data[:, :, :][mask.astype(bool)], axis=0)

            np.save(f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp_timecourse',
                    mask_mean
                    )

# =============================================================================
# Extract voxel wise timecourses
# =============================================================================

STIMDURS = [1, 2, 4, 12, 24]
# STIMDURS = [24]
EVENTDURS = np.array([11, 14, 20, 32, 48])
# EVENTDURS = np.array([48])

MODALITIES = ['vaso', 'bold']
# MODALITIES = ['bold']

for sub in subs:
    # =========================================================================
    # Look for sessions
    # Collectall runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))

    # Get dimensions
    nii = nb.load(allRuns[0])
    dims = nii.header['dim'][1:4]


    tr = findTR(f'code/stimulation/{sub}/ses-01/{sub}_ses-01_run-01_neurovascularCoupling.log')
    # print(f'Effective TR: {tr} seconds')
    tr = tr/UPFACTOR
    # print(f'Nominal TR will be: {tr} seconds')


    # Initialte list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1,6):  # We had a maximum of 2 sessions
            if f'ses-0{i}' in run:
                sessions.append(f'ses-0{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    for modality in MODALITIES:

        for stimDur, eventDur in zip(STIMDURS, EVENTDURS):
            dimsTmp = np.append(dims, int(eventDur/tr))
            tmp = np.zeros(dimsTmp)

            for ses in sessions:
                tmpRun = np.zeros(dimsTmp)

                # Set run name
                run = f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz'

                # Set design file of first run in session
                design = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-01_part-mag_bold_events.tsv')
                # Load run
                nii = nb.load(run)
                header = nii.header
                affine = nii.affine
                # Load data as array
                data = nii.get_fdata()

                mean = np.mean(data, axis = -1)

                tmpMean = np.zeros(data.shape)

                for i in range(tmpMean.shape[-1]):
                    tmpMean[...,i] = mean

                data = (np.divide(data, tmpMean) - 1) * 100

                onsets = design.loc[design['trial_type'] == f'stim {stimDur}s']

                for onset in onsets['onset']:
                    startTR = int(onset/tr)
                    endTR = startTR + int(eventDur/tr)
                    # test = data[..., startTR:endTR]

                    tmpRun += data[..., startTR:endTR]

                tmpRun /= len(onsets['onset'])
                # tmpRun = ((np.divide(tmp, tmpMean) - 1) * 100)

                tmp += tmpRun
                tmp /= len(sessions)

            img = nb.Nifti1Image(tmp, header = header, affine = affine)
            nb.save(img, f'{DATADIR}/{sub}/{sub}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDur}s.nii.gz')





# =============================================================================
# Upsample timecourses
# =============================================================================

# Define jitter
jitter = 4
# Divide by 2 because we already upsampled with a factor of 2 before
# factor = jitter/2

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
