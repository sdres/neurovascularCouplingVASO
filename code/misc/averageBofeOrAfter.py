"""Testing when to average functional data"""


import os
import glob
import nibabel as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# Get TR
UPFACTOR = 4

# =============================================================================
# Extract voxel wise time-courses
# =============================================================================

SUBS = ['sub-05']

STIMDURS = [1]

EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

# MODALITIES = ['vaso', 'bold']
MODALITIES = ['vaso']

for sub in SUBS:
    print(f'Working on {sub}')
    eraDir = f'{DATADIR}/{sub}/ERAs'  # Location of functional data

    # Make output folder if it does not exist already
    if not os.path.exists(eraDir):
        os.makedirs(eraDir)

    # =========================================================================
    # Look for sessions
    # Collect all runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))

    # Get dimensions
    nii = nb.load(allRuns[0])
    dims = nii.header['dim'][1:4]

    tr = 3.1262367768477795
    print(f'Effective TR: {tr} seconds')
    tr = tr/UPFACTOR
    print(f'Nominal TR will be: {tr} seconds')

    # Initiate list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1, 6):  # We had a maximum of 5 sessions
            if f'ses-0{i}' in run:
                sessions.append(f'ses-0{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    for modality in MODALITIES:
        print(f'\nProcessing {modality}')

        for j, stimDur in enumerate(STIMDURS):
            print(f'Processing stim duration: {stimDur}s')

            # Get number of volumes for stimulus duration epoch
            nrVols = int(EVENTDURS['longITI'][j]/tr)
            # Set dimensions of temporary data according to number of volumes
            dimsTmp = np.append(dims, nrVols)

            # Initialize empty data
            tmp = np.zeros(dimsTmp)
            divisor = np.zeros(dimsTmp)

            for ses in sessions:
                print(f'processing {ses}')

                # Set run name
                run = f'{DATADIR}/{sub}/{ses}/func/' \
                      f'{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz'

                # Set design file of first run in session
                design = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/'
                                     f'{sub}_{ses}_task-stimulation_run-01_part-mag_bold_events.tsv')
                # Load run
                nii = nb.load(run)
                header = nii.header
                affine = nii.affine
                # Load data as array
                data = nii.get_fdata()

                if data.shape[-1] > 900:
                    iti = 'longITI'
                else:
                    iti = 'shortITI'

                print(f'{sub} {ses} had {iti}')

                # Set baseline time
                baselineTime = 30  # in seconds

                # Compute baseline volumes
                baselineVols = int(baselineTime/tr)

                # Get volumes data
                baseline1 = data[..., :baselineVols]  # Initial baseline
                baseline2 = data[..., -baselineVols:]  # Ending baseline

                # Concatenate the two baselines
                baselineCombined = np.concatenate((baseline1, baseline2), axis=-1)

                # Compute mean across them
                baselineMean = np.mean(baselineCombined, axis=-1)

                # Prepare array for division
                tmpMean = np.zeros(data.shape)
                for i in range(tmpMean.shape[-1]):
                    tmpMean[..., i] = baselineMean

                # Actual signal normalization
                data = (np.divide(data, tmpMean) - 1) * 100

                # Get onsets of current stimulus duration
                onsets = design.loc[design['trial_type'] == f'stim {stimDur}s']

                # Make shape of run-average
                eventTimePoints = int(EVENTDURS[iti][j]/tr)

                runDims = np.append(dims, eventTimePoints)
                tmpRun = np.zeros(runDims)

                for onset in onsets['onset']:

                    startTR = int(onset/tr)
                    endTR = startTR + int(EVENTDURS[iti][j]/tr)

                    nrTimepoints = endTR - startTR
                    print(f'Extracting {nrTimepoints} timepoints')

                    tmpRun += data[..., startTR: endTR]

                tmpRun /= len(onsets['onset'])
                print('Adding session data to tmp')
                tmp[..., :nrTimepoints] += tmpRun

                # Add to divisor
                divisor[..., :nrTimepoints] += 1

                # print('Save session event-related average')
                # img = nb.Nifti1Image(tmpRun, header=header, affine=affine)
                # nb.save(img, f'{DATADIR}/{sub}/ERAs/'
                #              f'{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDur}s_sigChange.nii.gz')

            # Divide each time-point by the number of runs that went into it
            tmp = np.divide(tmp, divisor)

            # Change sign of VASO
            if modality == 'vaso':
                tmp = tmp * -1

            # Save event-related average across runs
            img = nb.Nifti1Image(tmp, header=header, affine=affine)
            nb.save(img, f'/Users/sebastiandresbach/'
                         f'before.nii.gz')


for sub in SUBS:
    print(f'Working on {sub}')
    eraDir = f'{DATADIR}/{sub}/ERAs'  # Location of functional data

    # Make output folder if it does not exist already
    if not os.path.exists(eraDir):
        os.makedirs(eraDir)

    # =========================================================================
    # Look for sessions
    # Collect all runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))

    # Get dimensions
    nii = nb.load(allRuns[0])
    dims = nii.header['dim'][1:4]

    tr = 3.1262367768477795
    print(f'Effective TR: {tr} seconds')
    tr = tr/UPFACTOR
    print(f'Nominal TR will be: {tr} seconds')

    # Initiate list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1, 6):  # We had a maximum of 5 sessions
            if f'ses-0{i}' in run:
                sessions.append(f'ses-0{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    for modality in MODALITIES:
        print(f'\nProcessing {modality}')

        for j, stimDur in enumerate(STIMDURS):
            print(f'Processing stim duration: {stimDur}s')

            # Get number of volumes for stimulus duration
            nrVols = int(EVENTDURS['longITI'][j]/tr)
            # Set dimensions of temporary data according to number of volumes
            dimsTmp = np.append(dims, nrVols)

            # Initialize empty data
            after = np.zeros(dimsTmp)
            before = np.zeros(dimsTmp)
            beforedivisor = np.zeros(dimsTmp)

            divisor = np.zeros(dimsTmp)

            # ================================================================================================
            # Initialize Baseline

            # Set baseline time
            baselineTime = 30  # in seconds
            # Compute baseline volumes
            baselineVols = int(baselineTime / tr)

            # Initialize baseline data
            baselineShape = np.append(dims, baselineVols*2)
            baselineData = np.zeros(baselineShape)

            for ses in sessions:
                print(f'processing {ses}')

                # Set run name
                run = f'{DATADIR}/{sub}/{ses}/func/' \
                      f'{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz'

                # Set design file of first run in session
                design = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/'
                                     f'{sub}_{ses}_task-stimulation_run-01_part-mag_bold_events.tsv')
                # Load run
                nii = nb.load(run)
                header = nii.header
                affine = nii.affine
                # Load data as array
                data = nii.get_fdata()

                if data.shape[-1] > 900:
                    iti = 'longITI'
                else:
                    iti = 'shortITI'

                print(f'{sub} {ses} had {iti}')

                # Get volumes data
                baseline1 = data[..., :baselineVols]  # Initial baseline
                baseline2 = data[..., -baselineVols:]  # Ending baseline

                # Concatenate the two baselines
                baselineCombined = np.concatenate((baseline1, baseline2), axis=-1)
                baselineData += baselineCombined  # Add baseline to overall baseline

                # # Compute mean across them
                baselineMean = np.mean(baselineCombined, axis=-1)

                # Get onsets of current stimulus duration
                onsets = design.loc[design['trial_type'] == f'stim {stimDur}s']

                # Make shape of run-average
                eventTimePoints = int(EVENTDURS[iti][j]/tr)
                runDims = np.append(dims, eventTimePoints)
                tmpRun = np.zeros(runDims)

                for onset in onsets['onset']:

                    startTR = int(onset/tr)
                    endTR = startTR + int(EVENTDURS[iti][j]/tr)

                    nrTimepoints = endTR - startTR

                    tmpRun += data[..., startTR: endTR]

                tmpRun /= len(onsets['onset'])

                print('Adding session data to tmp')
                after[..., :nrTimepoints] += tmpRun
                # Add to divisor
                divisor[..., :nrTimepoints] += 1

                # Compute signal change for event
                # Prepare array for division
                tmpMean = np.zeros(tmp[..., :nrTimepoints].shape)
                for i in range(tmpMean.shape[-1]):
                    tmpMean[..., i] = baselineMean

                # Actual signal normalization
                tmpRun = (np.divide(tmpRun, tmpMean) - 1) * 100

                # print('Save session event-related average')
                # img = nb.Nifti1Image(tmpRun, header=header, affine=affine)
                # nb.save(img, f'{DATADIR}/{sub}/ERAs/'
                #              f'{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDur}s_sigChange.nii.gz')

                before[..., :nrTimepoints] += tmpRun
                beforedivisor[..., :nrTimepoints] += 1

            before /= beforedivisor
            # Change sign of VASO
            if modality == 'vaso':
                before = before * -1
            # Save event-related average across runs
            img = nb.Nifti1Image(before, header=header, affine=affine)
            nb.save(img, f'/Users/sebastiandresbach/'
                         f'before2.nii.gz')
            # Divide each time-point by the number of runs that went into it
            after = np.divide(after, divisor)

            # Compute signal change of event
            baselineData /= len(sessions)
            baselineData = np.mean(baselineData, axis=-1)

            # Compute signal change for event
            # Prepare array for division
            tmpMean = np.zeros(tmp.shape)
            for i in range(tmpMean.shape[-1]):
                tmpMean[..., i] = baselineData

            # Actual signal normalization
            after = (np.divide(after, tmpMean) - 1) * 100

            # Change sign of VASO
            if modality == 'vaso':
                after = after * -1

            # Save event-related average across runs
            img = nb.Nifti1Image(after, header=header, affine=affine)
            nb.save(img, f'/Users/sebastiandresbach/'
                         f'after2.nii.gz')

