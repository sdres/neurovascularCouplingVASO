"""Response to Reviewer 2 - compare first to last trials of 24s stimulation across runs"""

import os
import glob
import datetime
import calendar
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import glob
import os
import nibabel as nb
import numpy as np
import re
import sys


# =======================================================================================
# Assess the duration between the first and last trial with 12 and 24 seconds stimuation
# =======================================================================================

subs = ['sub-05', 'sub-07', 'sub-09']

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

between_12 = []
between_24 = []

for sub in subs:
    # =========================================================================
    # Look for sessions
    # Collect all runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))

    # Initiate list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1, 6):  # We had a maximum of 5 sessions
            if f'ses-0{i}' in run:
                sessions.append(f'ses-0{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))

    for ses in sessions:
        # Set design file of first run in session
        design = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/'
                             f'{sub}_{ses}_task-stimulation_run-01_part-mag_bold_events.tsv')

        for j, stimDur in enumerate([12, 24]):
            tmp = design.loc[design['trial_type'] == f'stim {stimDur}s']
            between = tmp.onset.to_numpy()[-1] - tmp.onset.to_numpy()[0]

            if stimDur == 12:
                between_12.append(between)
            if stimDur == 24:
                between_24.append(between)

np.mean(between_12)
np.mean(between_24)

plt.hist(between_12, label='12')
plt.hist(between_24, label='24')
plt.legend()
plt.show()

# Concluding from this, we can say that on average, the 24 s trials are space further apart and we can go ahead


# =======================================================================================
# Temporally upsample first and last run per session
# =======================================================================================

# Define current dir
ROOT = os.getcwd()
sys.path.append('./code/misc')

from findTr import *

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
afniPath = '/Users/sebastiandresbach/abin'
layniiPath = '/Users/sebastiandresbach/git/laynii'

UPFACTOR = 4  # Must be an even number so that nulled and not-nulled timecourses can match

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
afniPath = '/Users/sebastiandresbach/abin'
antsPath = '/Users/sebastiandresbach/ANTs/install/bin'

# subs = ['sub-06']

for sub in subs:
    # Create subject-directory in derivatives if it does not exist
    subDir = f'{ROOT}/derivatives/{sub}'

    # =========================================================================
    # Look for sessions
    # Collect all runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))

    # Initiate list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1, 6):  # We had a maximum of 5 sessions
            if f'ses-0{i}' in run:
                sessions.append(f'ses-0{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))

    for ses in sessions:
        print(f'Processing {ses}')

        outFolder = f'{ROOT}/derivatives/{sub}/{ses}/func'

        tr = findTR(f'code/stimulation/{sub}/{ses}/{sub}_{ses}_run-01_neurovascularCoupling.log')
        tr = tr
        # print(f'Effective TR: {tr} seconds')
        tr = tr/UPFACTOR
        # print(f'Nominal TR will be: {tr} seconds')

        # Find nr runs
        sesRuns = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*run-0*_part-mag*bold.nii.gz'))

        nrRuns = len(sesRuns)

        nii = nb.load(sesRuns[0])
        header = nii.header
        affine = nii.affine
        # Load data as array
        data = nii.get_fdata()

        if data.shape[-1] > (900/4):
            iti = 'longITI'
            print(iti)
        else:
            iti = 'shortITI'
            print(f'{iti}, skipping')
            continue

        for runNr in [1, nrRuns]:
            for modality in ['bold']:
                # =====================================================================
                # Temporal upsampling
                # =====================================================================
                print(f'Temporally upsampling data with a factor of {UPFACTOR}.')
                command = f'{afniPath}/3dUpsample '
                command += f'-overwrite '
                command += f'-datum short '
                command += f'-prefix {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_intemp.nii.gz '
                command += f'-n {UPFACTOR} '
                command += f'-input {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_moco-reg.nii'
                subprocess.call(command, shell=True)

                # fix TR in header
                print('Fixing TR in header.')
                subprocess.call(
                    f'3drefit -TR {tr} '
                    + f'{outFolder}'
                    + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_intemp.nii.gz',
                    shell=True
                    )

                # ==================================================================
                # Multiply first BOLD timepoint to match timing between cbv and bold
                # ==================================================================
                if modality == 'bold':
                    print('Multiply first BOLD volume.')

                    # The number of volumes we have to prepend depends on the
                    # upsampling factor.
                    nrPrepend = int(UPFACTOR/2)

                    nii = nb.load(
                        f'{outFolder}'
                        + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_intemp.nii.gz'
                        )

                    # Load data
                    data = nii.get_fdata()  # Get data
                    header = nii.header  # Get header
                    affine = nii.affine  # Get affine

                    # Make new array
                    newData = np.zeros(data.shape)

                    for i in range(data.shape[-1]):
                        if i < nrPrepend:
                            newData[:, :, :, i] = data[:, :, :, 0]
                        else:
                            newData[:, :, :, i] = data[:, :, :, i-nrPrepend]

                    # Save data
                    img = nb.Nifti1Image(newData.astype(int), header=header, affine=affine)
                    nb.save(img, f'{outFolder}'
                        + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_intemp.nii.gz'
                        )


# =============================================================================
# Extract voxel wise time-courses per run
# =============================================================================

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# Get TR
UPFACTOR = 4


# SUBS = ['sub-06']

STIMDURS = [1, 2, 4, 12, 24]

EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

# MODALITIES = ['vaso', 'bold']
MODALITIES = ['bold']

for sub in subs:
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

    for modality in ['bold']:
        print(f'\nProcessing {modality}')

        for j, stimDur in enumerate([24]):
            print(f'Processing stim duration: {stimDur}s')

            # Get number of volumes for stimulus duration epoch
            nrVols = int(EVENTDURS['longITI'][-1]/tr)
            # Set dimensions of temporary data according to number of volumes
            dimsTmp = np.append(dims, nrVols)

            for ses in sessions:
                print(f'processing {ses}')

                # Find nr runs
                sesRuns = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*run-0*_part-mag*{modality}.nii.gz'))
                nrRuns = len(sesRuns)

                nii = nb.load(sesRuns[0])
                header = nii.header
                affine = nii.affine
                # Load data as array
                data = nii.get_fdata()

                if data.shape[-1] > (900/UPFACTOR):
                    iti = 'longITI'
                    print(iti)
                else:
                    iti = 'shortITI'
                    print(f'{iti}, skipping')
                    continue

                for runNr in [1, nrRuns]:

                    # Set run name
                    run = f'{DATADIR}/{sub}/{ses}/func/' \
                          f'{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_intemp.nii.gz'

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
                    eventTimePoints = int(EVENTDURS[iti][-1]/tr)

                    runDims = np.append(dims, eventTimePoints)
                    tmpRun = np.zeros(runDims)

                    for onset in onsets['onset']:

                        startTR = int(onset/tr)
                        endTR = startTR + int(EVENTDURS[iti][-1]/tr)

                        nrTimepoints = endTR - startTR
                        print(f'Extracting {nrTimepoints} timepoints')

                        tmpRun += data[..., startTR: endTR]

                    tmpRun /= len(onsets['onset'])

                    print(f'Save run-0{runNr} event-related average')
                    img = nb.Nifti1Image(tmpRun, header=header, affine=affine)
                    nb.save(img, f'{DATADIR}/{sub}/ERAs/'
                                 f'{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_intemp_era-{stimDur}s_sigChange.nii.gz')


# =============================================================================
# Average first and last run ERAs across sessions
# =============================================================================


ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# Get TR
UPFACTOR = 4


# MODALITIES = ['vaso', 'bold']
MODALITIES = ['bold']

for sub in subs:
    print(f'Working on {sub}')
    eraDir = f'{DATADIR}/{sub}/ERAs'  # Location of functional data

    for modality in ['bold']:
        print(f'\nProcessing {modality}')

        for j, stimDur in enumerate([24]):
            print(f'Processing stim duration: {stimDur}s')

            eras = sorted(glob.glob(f'{eraDir}/{sub}_ses-0*_task-stimulation_run-0*_part-mag_{modality}_intemp*'))

            # Average first runs
            for i, era in enumerate(eras[::2]):

                if i == 0:
                    # Load run
                    nii = nb.load(era)
                    print(era)
                    header = nii.header
                    affine = nii.affine
                    # Load data as array

                    data = nii.get_fdata()

                if i != 0:
                    # Load run
                    nii = nb.load(era)
                    print(era)
                    data += nii.get_fdata()

            data /= (i+1)
            img = nb.Nifti1Image(data, header=header, affine=affine)
            nb.save(img, f'{DATADIR}/{sub}/ERAs/'
                         f'{sub}_ses-avg_task-stimulation_run-avgFirst_part-mag_{modality}_intemp_era-{stimDur}s_sigChange.nii.gz')

            # Average last runs
            for i, era in enumerate(eras[1::2]):

                if i == 0:
                    # Load run
                    nii = nb.load(era)
                    print(era)
                    header = nii.header
                    affine = nii.affine
                    # Load data as array

                    data = nii.get_fdata()

                if i != 0:
                    # Load run
                    nii = nb.load(era)
                    print(era)
                    data += nii.get_fdata()

            data /= (i + 1)
            img = nb.Nifti1Image(data, header=header, affine=affine)
            nb.save(img, f'{DATADIR}/{sub}/ERAs/'
                         f'{sub}_ses-avg_task-stimulation_run-avgLast_part-mag_{modality}_intemp_era-{stimDur}s_sigChange.nii.gz')

# =============================================================================
# Bring ROI into native ERA space
# =============================================================================

for sub in subs:

    print(f'Processing {sub}')
    # Defining folders
    anatDir = f'{DATADIR}/{sub}/ses-01/anat'  # Location of anatomical data
    regFolder = f'{anatDir}/registrationFiles'  # Folder where output will be saved

    # Take care: fixed and moving are flipped
    moving = glob.glob(f'{anatDir}/upsample/{sub}_rim-LH_perimeter_chunk_uncrop.nii.gz')[0]

    fixed = f'{DATADIR}/{sub}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_T1w.nii'

    command = 'antsApplyTransforms '
    command += f'--interpolation multiLabel '
    command += f'-d 3 '
    command += f'-i {moving} '
    command += f'-r {fixed} '
    command += f'-t {regFolder}/registered1_1InverseWarp.nii.gz '
    # IMPORTANT: We take the inverse transform!!!
    command += f'-t [{regFolder}/registered1_0GenericAffine.mat, 1] '
    command += f'-o {moving.split(".")[0]}_registered.nii'

    subprocess.run(command, shell=True)


# =============================================================================
# Extract voxel wise time-course data
# =============================================================================

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# Get TR
UPFACTOR = 4

STIMDURS = [24]

EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

# MODALITIES = ['vaso', 'bold']
MODALITIES = ['bold']


subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
# SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-09']
# SUBS = ['sub-06']

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []
runOrderList = []
dataTypeList = []

for sub in subs:
    print('')
    print(sub)

    subDir = f'{DATADIR}/{sub}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    roisData = nb.load(glob.glob(f'{segFolder}/{sub}_rim-LH_perimeter_chunk_uncrop_registered.nii')[0]).get_fdata()
    roiIdx = roisData == 1

    for stimDuration in [24]:
    # for stimDuration in [1]:
        print(f'stimDur: {stimDuration}s')
        # for modality in ['vaso', 'bold']:
        for modality in ['bold']:

            print(modality)
            for position in ['First', 'Last']:
                era = sorted(glob.glob(f'{DATADIR}/{sub}/ERAs/{sub}_ses-avg_task-stimulation_run-avg{position}_part-mag_{modality}_'
                                          f'intemp_era-{stimDuration}s_sigChange.nii.gz'))[0]

                # Zscore data
                nii = nb.load(era)
                data = nii.get_fdata()
                dataShape = data.shape

                for volume in range(dataShape[-1]):

                    tmp = data[:, :, :, volume]
                    val = np.mean(tmp[roiIdx])

                    valList.append(val)
                    subList.append(sub)
                    stimDurList.append(stimDuration)
                    modalityList.append(modality)
                    timePointList.append(volume)
                    runOrderList.append(position)

data = pd.DataFrame({'subject': subList,
                     'volume': timePointList,
                     'modality': modalityList,
                     'data': valList,
                     'stimDur': stimDurList,
                     'runPosition': runOrderList})



sns.lineplot(data, x='volume', y='data', hue='runPosition')




data.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored.csv',
            sep=',',
            index=False)
#
# data.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_sub-08.csv',
#             sep=',',
#             index=False)
#
# data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored.csv', sep=',')
# data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_sub-08.csv', sep=',')

# Equalize mean between first and second set
tr = 3.1262367768477795/4
EVENTDURS = {'shortITI': (np.array([11, 14, 18, 32, 48])/tr).astype('int'),
             'longITI': (np.array([21, 24, 28, 42, 64])/tr).astype('int')}

STIMDURS = [1, 2, 4, 12, 24]

equalized = pd.DataFrame()

for sub in data['subject'].unique():
    for dataType in ['raw', 'zscore']:
        for modality in ['vaso', 'bold']:
        # for modality in ['vaso']:
                for layer in data['layer'].unique():
                for i, stimDur in enumerate(STIMDURS):
                    tmp = data.loc[(data['dataType'] == dataType)
                                   & (data['subject'] == sub)
                                   & (data['modality'] == modality)
                                   & (data['layer'] == layer)
                                   & (data['stimDur'] == stimDur)]

                    extension = EVENTDURS['longITI'][i] - EVENTDURS['shortITI'][i]
                    # Get max number of volumes
                    maxVol = np.max(tmp['volume'].to_numpy())

                    firstVol = maxVol - extension

                    series1 = tmp.loc[(tmp['volume'] < firstVol+1)]
                    series2 = tmp.loc[(tmp['volume'] >= firstVol+1)]

                    val1 = np.mean(series1.loc[series1['volume'] == firstVol]['data'])
                    val2 = np.mean(series2.loc[series2['volume'] == firstVol+1]['data'])

                    diff = val1 - val2
                    series2['data'] += diff

                    equalized = pd.concat((equalized, series1))
                    equalized = pd.concat((equalized, series2))

equalized.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalized.csv',
            sep=',',
            index=False)


equalized = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalized.csv', sep=',')

normalized = pd.DataFrame()

for sub in SUBS:
    for modality in ['bold', 'vaso']:
    # for modality in ['vaso']:
        for stimDuration in [1., 2., 4., 12., 24.]:
            for layer in [1, 2, 3]:
                for dataType in ['raw', 'zscore']:

                    tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                                        & (equalized['layer'] == layer)
                                        & (equalized['modality'] == modality)
                                        & (equalized['dataType'] == dataType)
                                        & (equalized['subject'] == sub)]


                    if dataType == 'raw':
                        # Get value of first volume for given layer
                        val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
                        # Normalize to that value
                        tmp['data'] = tmp['data'] - val

                    normalized = pd.concat((normalized, tmp))

normalized.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalizedNormalized.csv',
            sep=',',
            index=False)


# # =============================================================================
# # RETIRED CODE
# # =============================================================================


#
#
# # =============================================================================
# # Register ERAs to anatomy
# # =============================================================================
#
# subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
# subs = ['sub-06']
#
# # Define data dir
# DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'
#
# BBOX = {'sub-05': {'RH': {'xlower': 435, 'xrange': 162, 'ylower': 55, 'yrange': 162, 'zlower': 95, 'zrange': 158},
#                    'LH': {'xlower': 263, 'xrange': 162, 'ylower': 35, 'yrange': 162, 'zlower': 79, 'zrange': 158}},
#         'sub-06': {'LH': {'xlower': 271, 'xrange': 162, 'ylower': 7, 'yrange': 162, 'zlower': 31, 'zrange': 159}},
#         'sub-07': {'LH': {'xlower': 271, 'xrange': 166, 'ylower': 35, 'yrange': 158, 'zlower': 23, 'zrange': 166}},
#         'sub-08': {'LH': {'xlower': 275, 'xrange': 162, 'ylower': 15, 'yrange': 162, 'zlower': 47, 'zrange': 158}},
#         'sub-09': {'RH': {'xlower': 415, 'xrange': 162, 'ylower': 11, 'yrange': 162, 'zlower': 91, 'zrange': 158},
#                    'LH': {'xlower': 303, 'xrange': 162, 'ylower': 0, 'yrange': 162, 'zlower': 59, 'zrange': 158}}
#         }
#
# for sub in subs:
#
#     print(f'Processing {sub}')
#     # Defining folders
#     anatDir = f'{DATADIR}/{sub}/ses-01/anat'  # Location of anatomical data
#     regFolder = f'{anatDir}/registrationFiles'  # Folder where output will be saved
#
#     # =========================================================================
#     # register timeseries
#     # =========================================================================
#
#     eraDir = f'{DATADIR}/{sub}/ERAs'  # Location of functional data
#
#     for runPosition in ['First', 'Last']:
#         eras = sorted(glob.glob(f'{eraDir}/*avg{runPosition}*.nii.gz'))
#         outFolder = f'{eraDir}/frames'
#         tmpBox = BBOX[sub]['LH']
#
#         # Make output folder if it does not exist already
#         if not os.path.exists(outFolder):
#             os.makedirs(outFolder)
#
#         for era in eras:
#
#             basename = os.path.basename(era).split('.')[0]
#
#             # Split frames
#             nii = nb.load(era)
#             header = nii.header
#             affine = nii.affine
#             data = nii.get_fdata()
#
#             for i in range(data.shape[-1]):
#                 outName = f'{outFolder}/{basename}_frame{i:02d}.nii.gz'
#
#                 if os.path.exists(outName):
#                     print(f'file exists')
#                     continue
#
#                 frame = data[..., i]
#                 img = nb.Nifti1Image(frame, header=header, affine=affine)
#                 nb.save(img, outName)
#
#                 # # ==================================================================
#                 # # Mask with sphere
#                 # command = 'fslmaths '
#                 # command += f'{outName} '
#                 # command += f'-mul {anatDir}/upsample/{sub}_LH_sphere_ups4X_registered.nii '
#                 # command += f'{outName}'
#                 #
#                 # subprocess.run(command, shell=True)
#
#                 # ==================================================================
#                 # Apply inverse transform
#                 # Take care: fixed and moving are flipped
#                 fixed = glob.glob(f'{anatDir}/upsample/'
#                                   f'{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz')[0]
#
#                 moving = outName
#
#                 command = 'antsApplyTransforms '
#                 command += f'--interpolation BSpline[5] '
#                 command += f'-d 3 '
#                 command += f'-i {moving} '
#                 command += f'-r {fixed} '
#                 command += f'-t {regFolder}/registered1_1InverseWarp.nii.gz '
#                 # IMPORTANT: We take the inverse transform!!!
#                 command += f'-t [{regFolder}/registered1_0GenericAffine.mat, 1] '
#                 command += f'-o {moving.split(".")[0]}_registered.nii.gz'
#
#                 subprocess.run(command, shell=True)
#
#                 command = f'fslmaths {moving.split(".")[0]}_registered.nii.gz ' \
#                           f'-mul 1 ' \
#                           f'{moving.split(".")[0]}_registered.nii.gz ' \
#                           f'-odt float'
#
#                 subprocess.run(command, shell=True)
#
#                 # =========================================================================
#                 # Crop map
#                 inFile = f'{moving.split(".")[0]}_registered.nii.gz'
#                 base = inFile.split('.')[0]
#                 outFile = f'{base}_crop.nii.gz'
#
#                 command = 'fslroi '
#                 command += f'{inFile} '
#                 command += f'{outFile} '
#                 command += f"{tmpBox['xlower']} " \
#                            f"{tmpBox['xrange']} " \
#                            f"{tmpBox['ylower']} " \
#                            f"{tmpBox['yrange']} " \
#                            f"{tmpBox['zlower']} " \
#                            f"{tmpBox['zrange']}"
#
#                 subprocess.run(command, shell=True)
#
#                 # =========================================================================
#                 # Delete large registered file
#                 command = f'rm {inFile}'
#                 subprocess.run(command, shell=True)