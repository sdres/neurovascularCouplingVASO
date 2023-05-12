"""Get event-related averages per stimulus duration"""

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

SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-09']

STIMDURS = [1, 2, 4, 12, 24]

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

                print('Save session event-related average')
                img = nb.Nifti1Image(tmpRun, header=header, affine=affine)
                nb.save(img, f'{DATADIR}/{sub}/ERAs/'
                             f'{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDur}s_sigChange.nii.gz')

            # Divide each time-point by the number of runs that went into it
            tmp = np.divide(tmp, divisor)

            # Change sign of VASO
            if modality == 'vaso':
                tmp = tmp * -1

            # Save event-related average across runs
            img = nb.Nifti1Image(tmp, header=header, affine=affine)
            nb.save(img, f'{DATADIR}/{sub}/ERAs/'
                         f'{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDur}s_sigChange.nii.gz')


# =============================================================================
# Get data
# =============================================================================

SUBS = ['sub-09']

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []
depthList = []
sesLList = []

for sub in SUBS:
    subDir = f'{DATADIR}/{sub}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    depthFile = glob.glob(f'{segFolder}/3layers_layers_equivol.nii*')[0]
    depthNii = nb.load(depthFile)
    depthData = depthNii.get_fdata()
    layers = np.unique(depthData)[1:]

    roisData = nb.load(glob.glob(f'{segFolder}/{sub}_rim*_perimeter_chunk.nii*')[0]).get_fdata()
    roiIdx = roisData == 1

    for stimDuration in [1, 2, 4, 12, 24]:
        # for modality in ['bold']:
        for modality in ['vaso', 'bold']:

            for ses in ['ses-01', 'ses-03', 'ses-04', 'ses-05']:
                frames = sorted(glob.glob(f'{DATADIR}/{sub}/ERAs/frames/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_'
                                          f'intemp_era-{stimDuration}s_sigChange_frame*_registered_crop.nii.gz'))

                for j, frame in enumerate(frames):

                    nii = nb.load(frame)

                    data = nii.get_fdata()

                    for layer in layers:

                        layerIdx = depthData == layer
                        tmp = roiIdx*layerIdx

                        val = np.mean(data[tmp])

                        if modality == 'bold':
                            valList.append(val)
                        elif (modality == 'vaso'):
                            valList.append(-val)
                        # elif modality == 'vaso':
                        #     valList.append(val)

                        subList.append(sub)
                        depthList.append(layer)
                        stimDurList.append(stimDuration)
                        modalityList.append(modality)
                        timePointList.append(j)
                        sesLList.append(ses)

data = pd.DataFrame({'subject': subList,
                     'volume': timePointList,
                     'modality': modalityList,
                     'data': valList, 'layer': depthList,
                     'stimDur': stimDurList,
                    'session': sesLList})

# =============================================================================
# plot single subs
# =============================================================================

plt.style.use('dark_background')

palettesLayers = {'vaso': ['#55a8e2', '#aad4f0', '#ffffff', '#FF0000'],
                  'bold': ['#ff8c26', '#ffd4af', '#ffffff', '#FF0000']}

layerNames = ['deep', 'middle', 'superficial', 'vein']


for sub in ['sub-09']:

    for modality in ['bold', 'vaso']:
    # for modality in ['bold']:

        for stimDuration in [1., 2., 4., 12., 24.]:
            for ses in ['ses-01', 'ses-03', 'ses-04', 'ses-05']:

                fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

                for layer in [1, 2, 3]:

                    tmp = data.loc[(data['stimDur'] == stimDuration)
                                   & (data['layer'] == layer)
                                   & (data['modality'] == modality)
                                   & (data['session'] == ses)]

                    # Get number of volumes for stimulus duration
                    nrVols = len(np.unique(tmp['volume']))

                    # Get value of first volume for given layer
                    val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
                    # Normalize to that value
                    tmp['data'] = tmp['data'] - val

                    if modality == 'vaso':
                        ax1.set_ylim(-1.3, 5.1)
                    if modality == 'bold':
                        ax1.set_ylim(-3.1, 8.1)

                    sns.lineplot(ax=ax1,
                                 data=tmp,
                                 x="volume",
                                 y="data",
                                 color=palettesLayers[modality][layer-1],
                                 linewidth=3,
                                 # ci=None,
                                 label=layerNames[layer-1],
                                 )

                    # Set font-sizes for axes
                    ax1.yaxis.set_tick_params(labelsize=18)
                    ax1.xaxis.set_tick_params(labelsize=18)

                    # Tweak x-axis
                    ticks = np.linspace(0, nrVols, 10)
                    labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)
                    ax1.set_xticks(ticks[::2])
                    ax1.set_xticklabels(labels[::2], fontsize=18)
                    ax1.set_xlabel('Time [s]', fontsize=24)

                    # Draw stimulus duration
                    ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
                    # Draw line at 0% signal-change
                    ax1.axhline(0, linestyle='--', color='white')

                    # Prepare legend
                    legend = ax1.legend(loc='upper right', title="Layer", fontsize=18)
                    legend.get_title().set_fontsize('18')  # Legend 'Title' font-size

                    ax1.set_ylabel(r'Signal change [%]', fontsize=24)

                    # Set title
                    titlePad = 10
                    if stimDuration == 1:
                        plt.title(f'{int(stimDuration)} second stimulation {ses}', fontsize=24, pad=titlePad)
                    else:
                        plt.title(f'{int(stimDuration)} seconds stimulation {ses}', fontsize=24, pad=titlePad)

                    plt.tight_layout()
                    # plt.savefig(f'./results/ERAs/{sub}_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers.png',
                    #             bbox_inches="tight")
                    plt.show()


# =============================================================================
# Baseline after averaging
# =============================================================================

SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']

STIMDURS = [1, 2, 4, 12, 24]

EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

# MODALITIES = ['vaso', 'bold']
MODALITIES = ['bold']

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
            tmp = np.zeros(dimsTmp)
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

                # Compute mean across them
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
                tmp[..., :nrTimepoints] += tmpRun
                # Add to divisor
                divisor[..., :nrTimepoints] += 1

                # Compute signal change for event
                # Prepare array for division
                # tmpMean = np.zeros(tmp[..., :nrTimepoints].shape)
                # for i in range(tmpMean.shape[-1]):
                #     tmpMean[..., i] = baselineMean

                # Actual signal normalization
                # tmpRun = (np.divide(tmpRun, tmpMean) - 1) * 100

                # print('Save session event-related average')
                # img = nb.Nifti1Image(tmpRun, header=header, affine=affine)
                # nb.save(img, f'{DATADIR}/{sub}/ERAs/'
                #              f'{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDur}s_sigChange.nii.gz')

            # Divide each time-point by the number of runs that went into it
            tmp = np.divide(tmp, divisor)

            # Compute signal change of event
            baselineData /= len(sessions)
            baselineData = np.mean(baselineData, axis=-1)

            # Compute signal change for event
            # Prepare array for division
            tmpMean = np.zeros(tmp.shape)
            for i in range(tmpMean.shape[-1]):
                tmpMean[..., i] = baselineData

            # Actual signal normalization
            tmp = (np.divide(tmp, tmpMean) - 1) * 100

            # Change sign of VASO
            if modality == 'vaso':
                tmp = tmp * -1

            # Save event-related average across runs
            img = nb.Nifti1Image(tmp, header=header, affine=affine)
            nb.save(img, f'{DATADIR}/{sub}/ERAs/'
                         f'{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDur}s_sigChange-after.nii.gz')


# =============================================================================
# Get data
# =============================================================================

SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-09']
# SUBS = ['sub-05']

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []
depthList = []
dataTypeList = []

for sub in SUBS:
    print('')
    print(sub)

    subDir = f'{DATADIR}/{sub}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    depthFile = glob.glob(f'{segFolder}/3layers_layers_equivol.nii*')[0]
    depthNii = nb.load(depthFile)
    depthData = depthNii.get_fdata()
    layers = np.unique(depthData)[1:]

    roisData = nb.load(glob.glob(f'{segFolder}/{sub}_rim-LH_perimeter_chunk.nii*')[0]).get_fdata()
    roiIdx = roisData == 1

    for stimDuration in [1, 2, 4, 12, 24]:
        print(f'stimDur: {stimDuration}s')
        for modality in ['vaso', 'bold']:
            print(modality)
            frames = sorted(glob.glob(f'{DATADIR}/{sub}/ERAs/frames/{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}_'
                                      f'intemp_era-{stimDuration}s_sigChange-after_frame*_registered_crop.nii.gz'))

            # print(f'Found {len(frames)} timepoints')
            # Zscore data
            nii = nb.load(frames[0])
            dataShape = nii.get_fdata().shape
            dataShape = np.append(dataShape, len(frames))
            data = np.zeros(dataShape)

            for n in range(len(frames)):
                nii = nb.load(frames[n])
                tmp = nii.get_fdata()
                data[..., n] = tmp

            mean = np.mean(data, axis=-1)
            stdDev = np.std(data, axis=-1)

            for j, frame in enumerate(frames):

                nii = nb.load(frame)
                data = nii.get_fdata()

                zscored = (data - mean)/stdDev

                for layer in layers:

                    layerIdx = depthData == layer
                    tmp = roiIdx*layerIdx

                    for dataType in ['raw', 'zscore']:
                        if dataType == 'raw':
                            val = np.mean(data[tmp])
                        if dataType == 'zscore':
                            val = np.mean(zscored[tmp])

                        valList.append(val)
                        subList.append(sub)
                        depthList.append(layer)
                        stimDurList.append(stimDuration)
                        modalityList.append(modality)
                        timePointList.append(j)
                        dataTypeList.append(dataType)

data = pd.DataFrame({'subject': subList,
                     'volume': timePointList,
                     'modality': modalityList,
                     'data': valList,
                     'layer': depthList,
                     'stimDur': stimDurList,
                     'dataType': dataTypeList})

data.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored.csv',
            sep=',',
            index=False)

data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored.csv', sep=',')

# Equalize mean between first and second set
tr = 3.1262367768477795/4
EVENTDURS = {'shortITI': (np.array([11, 14, 18, 32, 48])/tr).astype('int'),
             'longITI': (np.array([21, 24, 28, 42, 64])/tr).astype('int')}

STIMDURS = [1, 2, 4, 12, 24]

equalized = pd.DataFrame()

for sub in data['subject'].unique():
    for dataType in ['raw', 'zscore']:
        for modality in ['vaso', 'bold']:
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


# =============================================================================
# plot single subs
# =============================================================================

plt.style.use('dark_background')

palettesLayers = {'vaso': ['#55a8e2', '#aad4f0', '#ffffff', '#FF0000'],
                  'bold': ['#ff8c26', '#ffd4af', '#ffffff', '#FF0000']}

layerNames = ['deep', 'middle', 'superficial', 'vein']

# for sub in ['sub-06', 'sub-07', 'sub-09']:
for sub in ['sub-06']:

    # for modality in ['bold', 'vaso']:
    for modality in ['vaso']:

        # for stimDuration in [1., 2., 4., 12., 24.]:
        for stimDuration in [24]:

            fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

            for layer in [1, 2, 3]:

                tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                               & (equalized['layer'] == layer)
                               & (equalized['modality'] == modality)
                               & (equalized['subject'] == sub)
                               & (equalized['dataType'] == 'raw')]
                #
                # tmp = data.loc[(data['stimDur'] == stimDuration)
                #                & (data['layer'] == layer)
                #                & (data['modality'] == modality)
                #                & (data['subject'] == sub)
                #                & (data['dataType'] == 'raw')]

                # Get number of volumes for stimulus duration
                nrVols = len(np.unique(tmp['volume']))

                # Get value of first volume for given layer
                val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
                # Normalize to that value
                tmp['data'] = tmp['data'] - val

                if modality == 'vaso':
                    ax1.set_ylim(-2.1, 5.1)
                if modality == 'bold':
                    ax1.set_ylim(-3.1, 7.1)

                sns.lineplot(ax=ax1,
                             data=tmp,
                             x="volume",
                             y="data",
                             color=palettesLayers[modality][layer-1],
                             linewidth=3,
                             # ci=None,
                             label=layerNames[layer-1],
                             )

            # Set font-sizes for axes
            ax1.yaxis.set_tick_params(labelsize=18)
            ax1.xaxis.set_tick_params(labelsize=18)

            # Tweak x-axis
            # ticks = np.linspace(0, nrVols, 10)
            # labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)
            # ax1.set_xticks(ticks[::2])
            # ax1.set_xticklabels(labels[::2], fontsize=18)
            # ax1.set_xlabel('Time [s]', fontsize=24)

            # Draw stimulus duration
            ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
            # Draw line at 0% signal-change
            ax1.axhline(0, linestyle='--', color='white')

            # Prepare legend
            legend = ax1.legend(loc='upper right', title="Layer", fontsize=18)
            legend.get_title().set_fontsize('18')  # Legend 'Title' font-size

            ax1.set_ylabel(r'Signal change [%]', fontsize=24)

            # Set title
            titlePad = 10
            if stimDuration == 1:
                plt.title(f'{int(stimDuration)} second stimulation', fontsize=24, pad=titlePad)
            else:
                plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24, pad=titlePad)

            plt.tight_layout()
            # plt.savefig(f'./results/ERAs/{sub}_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers_old.png',
            #             bbox_inches="tight")
            plt.show()


for modality in ['bold', 'vaso']:
# for modality in ['bold']:

    for stimDuration in [1., 2., 4., 12., 24.]:
    # for stimDuration in [1]:

        fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

        for layer in [1, 2, 3]:

            tmp = data.loc[(data['stimDur'] == stimDuration)
                           & (data['layer'] == layer)
                           & (data['modality'] == modality)
                           & (data['dataType'] == 'raw')]

            tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                                & (equalized['layer'] == layer)
                                & (equalized['modality'] == modality)
                                & (equalized['dataType'] == 'raw')]

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))

            # Get value of first volume for given layer
            val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # Normalize to that value
            tmp['data'] = tmp['data'] - val

            if modality == 'vaso':
                ax1.set_ylim(-1.1, 5.1)
            if modality == 'bold':
                ax1.set_ylim(-3.1, 7.1)

            sns.lineplot(ax=ax1,
                         data=tmp,
                         x="volume",
                         y="data",
                         color=palettesLayers[modality][layer-1],
                         linewidth=3,
                         # ci=None,
                         label=layerNames[layer-1],
                         )

        # Set font-sizes for axes
        ax1.yaxis.set_tick_params(labelsize=18)
        ax1.xaxis.set_tick_params(labelsize=18)

        # Tweak x-axis
        ticks = np.linspace(0, nrVols, 10)
        labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)
        ax1.set_xticks(ticks[::2])
        ax1.set_xticklabels(labels[::2], fontsize=18)
        ax1.set_xlabel('Time [s]', fontsize=24)

        # Draw stimulus duration
        ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
        # Draw line at 0% signal-change
        ax1.axhline(0, linestyle='--', color='white')

        # Prepare legend
        legend = ax1.legend(loc='upper right', title="Layer", fontsize=18)
        legend.get_title().set_fontsize('18')  # Legend 'Title' font-size

        ax1.set_ylabel(r'Signal change [%]', fontsize=24)

        # Set title
        titlePad = 10
        if stimDuration == 1:
            plt.title(f'{int(stimDuration)} second stimulation', fontsize=24, pad=titlePad)
        else:
            plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24, pad=titlePad)

        plt.tight_layout()

        plt.savefig(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/group_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers.png',
                    bbox_inches="tight")
        plt.show()


# =============================================================================
# plot zscored ERAs
# =============================================================================

plt.style.use('dark_background')

palettesLayers = {'vaso': ['#55a8e2', '#aad4f0', '#ffffff', '#FF0000'],
                  'bold': ['#ff8c26', '#ffd4af', '#ffffff', '#FF0000']}

layerNames = ['deep', 'middle', 'superficial', 'vein']

for sub in ['sub-06', 'sub-07', 'sub-09']:
# for sub in ['sub-05']:

    for modality in ['bold', 'vaso']:
    # for modality in ['vaso']:

        for stimDuration in [1., 2., 4., 12., 24.]:
        # for stimDuration in [1]:

            fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

            for layer in [1, 2, 3]:

                tmp = data.loc[(data['stimDur'] == stimDuration)
                               & (data['layer'] == layer)
                               & (data['modality'] == modality)
                               & (data['subject'] == sub)
                               & (data['dataType'] == 'zscore')]

                # Get number of volumes for stimulus duration
                nrVols = len(np.unique(tmp['volume']))

                # Get value of first volume for given layer
                # val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
                # Normalize to that value
                # tmp['data'] = tmp['data'] - val

                # if modality == 'vaso':
                ax1.set_ylim(-1.1, 1.6)
                # if modality == 'bold':
                #     ax1.set_ylim(-3.1, 7.1)

                sns.lineplot(ax=ax1,
                             data=tmp,
                             x="volume",
                             y="data",
                             color=palettesLayers[modality][layer-1],
                             linewidth=3,
                             # ci=None,
                             label=layerNames[layer-1],
                             )

            # Set font-sizes for axes
            ax1.yaxis.set_tick_params(labelsize=18)
            ax1.xaxis.set_tick_params(labelsize=18)

            # Tweak x-axis
            ticks = np.linspace(0, nrVols, 10)
            labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)
            ax1.set_xticks(ticks[::2])
            ax1.set_xticklabels(labels[::2], fontsize=18)
            ax1.set_xlabel('Time [s]', fontsize=24)

            # Draw stimulus duration
            ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
            # Draw line at 0% signal-change
            ax1.axhline(0, linestyle='--', color='white')

            # Prepare legend
            legend = ax1.legend(loc='upper right', title="Layer", fontsize=18)
            legend.get_title().set_fontsize('18')  # Legend 'Title' font-size

            ax1.set_ylabel(r'Signal change [%, z-scored]', fontsize=24)

            # Set title
            titlePad = 10
            if stimDuration == 1:
                plt.title(f'{int(stimDuration)} second stimulation', fontsize=24, pad=titlePad)
            else:
                plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24, pad=titlePad)

            plt.tight_layout()
            plt.savefig(f'./results/ERAs/{sub}_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers_zscored.png',
                        bbox_inches="tight")
            plt.show()



for modality in ['bold', 'vaso']:
# for modality in ['vaso']:

    for stimDuration in [1., 2., 4., 12., 24.]:
    # for stimDuration in [1]:

        fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

        for layer in [1, 2, 3]:

            tmp = data.loc[(data['stimDur'] == stimDuration)
                           & (data['layer'] == layer)
                           & (data['modality'] == modality)
                           & (data['dataType'] == 'zscore')]


            tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                                & (equalized['layer'] == layer)
                                & (equalized['modality'] == modality)
                                & (equalized['dataType'] == 'zscore')]

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))

            # # Get value of first volume for given layer
            val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # # Normalize to that value
            # tmp['data'] -= val

            ax1.set_ylim(-1.1, 1.6)

            sns.lineplot(ax=ax1,
                         data=tmp,
                         x="volume",
                         y="data",
                         color=palettesLayers[modality][layer-1],
                         linewidth=3,
                         # ci=None,
                         label=layerNames[layer-1],
                         )

        # Set font-sizes for axes
        ax1.yaxis.set_tick_params(labelsize=18)
        ax1.xaxis.set_tick_params(labelsize=18)

        # Tweak x-axis
        ticks = np.linspace(0, nrVols, 10)
        labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)
        ax1.set_xticks(ticks[::2])
        ax1.set_xticklabels(labels[::2], fontsize=18)
        ax1.set_xlabel('Time [s]', fontsize=24)

        # Draw stimulus duration
        ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
        # Draw line at 0% signal-change
        # ax1.axhline(0, linestyle='--', color='white')

        # Prepare legend
        legend = ax1.legend(loc='upper right', title="Layer", fontsize=18)
        legend.get_title().set_fontsize('18')  # Legend 'Title' font-size

        ax1.set_ylabel(r'Signal change [%, z-scored]', fontsize=24)

        # Set title
        titlePad = 10
        if stimDuration == 1:
            plt.title(f'{int(stimDuration)} second stimulation', fontsize=24, pad=titlePad)
        else:
            plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24, pad=titlePad)

        plt.tight_layout()
        plt.savefig(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/group_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers_zscored.png',
                    bbox_inches="tight")
        plt.show()


# =====================================================================================================================
# Plot ERA per layer
# =====================================================================================================================

# ==================================================================
# set general styles

# Define figzize
FS = (8, 5)
# define linewidth to 2
LW = 2
# Define fontsize size for x- and y-labels
labelSize = 24
# Define fontsize size for x- and y-ticks
tickLabelSize = 18
# Define fontsize legend text
legendTextSize = 18
palettes = {
    'bold': ['#ff7f0e', '#ff9436', '#ffaa5e', '#ffbf86', '#ffd4af'],
    'vaso': ['#1f77b4', '#2a92da', '#55a8e2', '#7fbee9', '#aad4f0']}

for j, modality in enumerate(['bold', 'vaso']):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.subplots_adjust(top=0.8)

    for i, layer in enumerate(['deep', 'middle', 'superficial']):

        for k, stimDur in enumerate(data['stimDur'].unique()):
            # tmp = data.loc[(data['modality'] == modality)
            #                & (data['layer'] == i+1)
            #                & (data['stimDur'] == stimDur)
            #                & (data['dataType'] == 'raw')]
            tmp = equalized.loc[(equalized['modality'] == modality)
                           & (equalized['layer'] == i+1)
                           & (equalized['stimDur'] == stimDur)
                           & (equalized['dataType'] == 'raw')]


            # # Get value of first volume for given layer
            val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # # Normalize to that value
            tmp['data'] = tmp['data'] - val

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))
            sns.lineplot(ax=axes[i],
                         data=tmp,
                         x='volume',
                         y='data',
                         hue='stimDur',
                         linewidth=LW,
                         palette=[palettes[modality][k]],
                         errorbar=None)

        # ================================================================================
        # Tweak x-axis
        ticks = np.linspace(0, nrVols, 6)
        labels = (np.linspace(0, nrVols, 6) * 0.7808410714285715).astype('int')
        axes[i].set_xticks(ticks)
        axes[i].set_xticklabels(labels, fontsize=18)
        if i == 1:
            axes[i].set_xlabel('Time [s]', fontsize=24)
        else:
            axes[i].set_xlabel('', fontsize=24)

        # ================================================================================
        # Tweak y-axis
        if modality == 'vaso':
            axes[i].set_ylim(-1.1, 4.1)
        if modality == 'bold':
            axes[i].set_ylim(-3.3, 6.1)

        lim = axes[0].get_ylim()
        tickMarks = np.arange(lim[0].round(), lim[1], 1).astype('int')

        if i == 0:
            axes[i].set_ylabel(r'Signal change [%]', fontsize=labelSize)
            axes[i].set_yticks(tickMarks, tickMarks, fontsize=18)
        # elif i > 1:
        #     axes[i].set_ylabel(r'', fontsize=labelSize)
        #     axes[i].set_yticks([])

        # Set font-sizes for axes
        axes[i].yaxis.set_tick_params(labelsize=18)
        axes[i].xaxis.set_tick_params(labelsize=18)

        # ================================================================================
        # Misc
        axes[i].set_title(layer, fontsize=labelSize)
        # Draw lines
        axes[i].axhline(0, linestyle='--', color='white')
        # Legend
        if i < 3:
            axes[i].get_legend().remove()
        legend = axes[2].legend(title='Stim dur [s]', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        title = legend.get_title()
        title.set_fontsize(18)

    # plt.suptitle(f'{modality}', fontsize=labelSize, y=0.98)
    plt.tight_layout()
    plt.savefig(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/group_byLayer_{modality}_ERA.png',
                bbox_inches="tight")
    plt.show()


# Zoomed in
for j, modality in enumerate(['bold', 'vaso']):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.subplots_adjust(top=0.8)

    for i, layer in enumerate(['deep', 'middle', 'superficial']):

        for k, stimDur in enumerate(data['stimDur'].unique()):
            tmp = data.loc[(data['modality'] == modality)
                           & (data['layer'] == i + 1)
                           & (data['stimDur'] == stimDur)
                           & (data['dataType'] == 'raw')
                           & (data['volume'] <= 6)]

            # # Get value of first volume for given layer
            val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # # Normalize to that value
            tmp['data'] = tmp['data'] - val

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))
            sns.lineplot(ax=axes[i],
                         data=tmp,
                         x='volume',
                         y='data',
                         hue='stimDur',
                         linewidth=LW,
                         palette=[palettes[modality][k]],
                         errorbar=None)
        #
        # # ================================================================================
        # # Tweak x-axis
        # ticks = np.linspace(0, nrVols, 6)
        # labels = (np.linspace(0, nrVols, 6) * 0.7808410714285715).astype('int')
        # axes[i].set_xticks(ticks)
        # axes[i].set_xticklabels(labels, fontsize=18)
        # if i == 1:
        #     axes[i].set_xlabel('Time [s]', fontsize=24)
        # else:
        #     axes[i].set_xlabel('', fontsize=24)
        #
        # # ================================================================================
        # # Tweak y-axis
        # if modality == 'vaso':
        #     axes[i].set_ylim(-1.1, 4.1)
        # if modality == 'bold':
        #     axes[i].set_ylim(-3.3, 6.1)
        #
        # lim = axes[0].get_ylim()
        # tickMarks = np.arange(lim[0].round(), lim[1], 1).astype('int')
        #
        # if i == 0:
        #     axes[i].set_ylabel(r'Signal change [%]', fontsize=labelSize)
        #     axes[i].set_yticks(tickMarks, tickMarks, fontsize=18)
        # # elif i > 1:
        # #     axes[i].set_ylabel(r'', fontsize=labelSize)
        # #     axes[i].set_yticks([])
        #
        # # Set font-sizes for axes
        # axes[i].yaxis.set_tick_params(labelsize=18)
        # axes[i].xaxis.set_tick_params(labelsize=18)
        #
        # # ================================================================================
        # # Misc
        # axes[i].set_title(layer, fontsize=labelSize)
        # # Draw lines
        # axes[i].axhline(0, linestyle='--', color='white')
        # # Legend
        # if i < 3:
        #     axes[i].get_legend().remove()
        # legend = axes[2].legend(title='Stim dur [s]', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        # title = legend.get_title()
        # title.set_fontsize(18)

    # plt.suptitle(f'{modality}', fontsize=labelSize, y=0.98)
    plt.tight_layout()
    # plt.savefig(
    #     f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/group_byLayer_{modality}_ERA.png',
    #     bbox_inches="tight")
    plt.show()