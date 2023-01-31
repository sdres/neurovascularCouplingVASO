'''

Extracting data without interpolation

'''

import nibabel as nb
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from itertools import permutations
from scipy import signal
import re
import math

# Set derivatives directory
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# Set participants to work on
subs = ['sub-05','sub-06','sub-07','sub-08','sub-09']
subs = ['sub-06']

# Rearrange jitters to match sampling
boldJitters = {0.0: 1.570, 0.785: 0.785, 1.57: 0, 2.355: 2.355}
cbvJitters = {0.0: 0.0, 0.785: 2.355, 1.57: 1.570, 2.355: 0.785}

# Set event durations for runs with short and long ITIs
EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

tr = 0.785  # Set TR in seconds

# Define function to truncate values after decimals
def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

# ==============================================================================
# Extract data from mask
# ==============================================================================

# Initiate lists to store data
timepointList = []
runList = []
valList = []
stimDurList = []
modalityList = []

for sub in subs:  # Loop over participants

    # Set and load mask data
    maskFile = f'{ROOT}/{sub}/v1Mask.nii.gz'
    maskData = nb.load(maskFile).get_fdata()

    # for modality in ['bold', 'cbv']:
    for modality in ['cbv']:

        runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-stimulation_run-avg_part-mag_{modality}.nii'))

        for runFile in runs:
            base = os.path.basename(runFile).rsplit('.', 2)[0]
            print(f'Processing {base}')
            for i in range(1,99):
                if f'ses-0{i}' in base:
                    ses = f'ses-0{i}'

            log = f'code/stimulation/{sub}/{ses}/{sub}_{ses}_run-01_neurovascularCoupling.log'
            logFile = pd.read_csv(log,usecols=[0])

            # Because the column definition will get hickups if empty colums are
            # present, we find line with first trigger to then load the file anew,
            # starting with that line
            for index, row in logFile.iterrows():
                if re.search('Keypress: 5', str(row)):
                    firstVolRow = index
                    break

            # Define column names
            ColNames = ['startTime', 'type', 'event']

            # load logfile again, starting with first trigger
            logFile = pd.read_csv(log, sep = '\t',skiprows=firstVolRow, names = ColNames)

            runNii = nb.load(runFile)
            runData = runNii.get_fdata()

            # See if run has long or sort ITIs based on #volumes
            if runData.shape[-1] > 250:
                iti = 'longITI'
            if runData.shape[-1] < 250:
                iti = 'shortITI'

            # Linearly detrend data if wanted
            signal.detrend(runData, axis = -1, type = 'linear', overwrite_data = True)

            # Average timecouorse over entire mask
            timecourse = np.mean(runData[:, :, :][maskData.astype(bool)], axis=0)

            # Set baseline for normalization
            baseline = np.mean(np.concatenate((timecourse[:11],timecourse[-21:])), axis = -1)

            # Compute signal change with reference to baseline
            sigChange = (np.divide(timecourse, baseline)) * 100
            sigChange = (np.divide(timecourse, baseline)) * 100
            # sigChange = timecourse.copy()
            # sigChange = (np.divide(timecourse, baseline)-1) * 100

            for k, stimDur in enumerate([1,2,4,12,24]):

                # Initiate lists
                starts = []
                jitters = []

                stimSwitch = False

                # Loop over lines of log and find stimulation start and stop times
                for index, row in logFile.iterrows():
                    if not logFile['event'][index] != logFile['event'][index]:

                        if re.search('stimDur', logFile['event'][index]):
                            currStimDur = int(float(re.findall(r"\d+\.\d+", logFile['event'][index])[0]))

                            if currStimDur == stimDur:
                                stimSwitch = True

                        if stimSwitch:
                            if re.search('jitter', logFile['event'][index]):
                                currJitter = float(re.findall(r"\d+\.\d+", logFile['event'][index])[0])
                                jitters.append(currJitter)

                            if modality == 'bold':
                                if re.search('TR2', logFile['event'][index]):
                                    if currJitter <= 2:
                                        boldVol = int(re.findall(r"\d+", logFile['event'][index])[-1])  # because of 0-indexing of data
                                        starts.append(boldVol)
                                        stimSwitch = False
                                    if currJitter >= 2:
                                        boldVol = int(re.findall(r"\d+", logFile['event'][index])[-1])+1  # because of 0-indexing of data
                                        starts.append(boldVol)
                                        stimSwitch = False
                            if modality == 'cbv':
                                if re.search('TR1', logFile['event'][index]):
                                    if currJitter <= 0.5:
                                        cbvVol = int(re.findall(r"\d+", logFile['event'][index])[-1])  # because of 0-indexing of data
                                        starts.append(cbvVol)
                                        stimSwitch = False
                                    if currJitter >= 0.5:
                                        cbvVol = int(re.findall(r"\d+", logFile['event'][index])[-1])+1  # because of 0-indexing of data
                                        starts.append(cbvVol)
                                        stimSwitch = False

                if modality == 'bold':
                    jitters = [boldJitters[truncate(i, 3)] for i in jitters]
                if modality == 'cbv':
                    jitters = [cbvJitters[truncate(i, 3)] for i in jitters]

                length = int(np.round(EVENTDURS[iti][k]/(tr*4)))+1

                for i, start in enumerate(starts):

                    tmp = sigChange[start:start+length]

                    newStart = int(np.round(jitters[i]/tr))

                    for j, item in enumerate(tmp):
                        timepoint = int(newStart + int(j*4))
                        timepointList.append(timepoint)
                        valList.append(item)
                        runList.append(base)
                        stimDurList.append(stimDur)
                        modalityList.append(modality)



data = pd.DataFrame({'run':runList, 'val': valList, 'volume':timepointList, 'duration': stimDurList, 'modality': modalityList})
outName = './results/highresTimecourse.csv'
data.to_csv(outName, index = False)


data = pd.read_csv(outName)
noOutliers = data.copy()
outlierCount = 0
totalCount = 0

volumes = np.sort(data['volume'].unique())

# for modality in ['bold', 'cbv']:
for modality in ['cbv']:
    for k, stimDur in enumerate([1,2,4,12,24]):
        for vol in volumes:
            tmp = noOutliers.loc[(noOutliers['volume']==vol)&(noOutliers['duration']==stimDur)&(noOutliers['modality']==modality)]
            tmpVals = np.asarray(tmp['val'])

            percentile25 = tmp['val'].quantile(0.25)
            percentile75 = tmp['val'].quantile(0.75)

            iqr = percentile75-percentile25

            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr

            for val in tmpVals:
                totalCount += 1
                if val < lower_limit:
                    noOutliers.drop(noOutliers.loc[noOutliers['val']==val].index, inplace = True)
                    outlierCount += 1
                    print(stimDur)
            for val in tmpVals:
                if val > upper_limit:
                    noOutliers.drop(noOutliers.loc[data['val']==val].index, inplace = True)
                    outlierCount += 1
                    print(stimDur)


print(f'{outlierCount} out of {totalCount} datapoints were outliers')


plt.style.use('dark_background')
data = pd.read_csv(outName)
# for modality in ['bold', 'cbv']:
for modality in ['cbv']:
    for k, stimDur in enumerate([1,2,4,12,24]):
        # tmp = noOutliers.loc[(noOutliers['duration']==stimDur)&(noOutliers['modality']==modality)]
        tmp = data.loc[(data['duration']==stimDur)&(data['modality']==modality)]
        maxVol = np.amax(data['volume'])
        fig, ax = plt.subplots(1,1,figsize=(7.5,5))

        ax.axvspan(0, stimDur / tr, color='#e5e5e5', alpha=0.2, lw=0)


        sns.scatterplot(data=tmp, x="volume", y="val", hue='run', legend=False)
        # sns.scatterplot(data=tmp, x="volume", y="val", hue='run')
        # Plot mean
        sns.lineplot(data = tmp,
                     x = 'volume',
                     y = 'val',
                     linewidth = 3,
                     # color = '#ff8c26'
                     color = 'black'
                     # color = 'white'
                     )

        ax.set_ylabel('Normalized intensity', fontsize=24)

        # Prepare ticks for x axis
        xticks = np.linspace(0, maxVol, 8)
        xlabels = (xticks * tr).round(decimals = 1)

        ax.set_xticks(xticks, xlabels.astype('int'), fontsize = 18)
        ax.set_xlabel('Time [s]', fontsize=24)

        ax.yaxis.set_tick_params(labelsize=18)
        ax.xaxis.set_tick_params(labelsize=18)

        ax.axhline(100,linestyle = '--', color = 'white')


        if modality == 'cbv':
            titleModality = 'nulled'
        if modality == 'bold':
            titleModality = 'bold'
        plt.ylim(96,107)


        plt.title(f'{titleModality} {stimDur}s second stimulation', fontsize=24,pad=10)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.savefig(f'./results/{modality}_{stimDur}_noOutliers.png', bbox_inches = "tight")
        plt.savefig(f'./results/{modality}_{stimDur}.png', bbox_inches = "tight")
        plt.show()



for k, stimDur in enumerate([1,2,4,12,24]):
    cbvMean = np.zeros(len(data['volume'].unique()))
    volumes = np.sort(data['volume'].unique())


    maxVol = np.amax(data.loc[(data['modality']=='cbv')&(data['duration']==24)]['volume'])

    for i, vol in enumerate(volumes):
        tmp = data.loc[(data['volume']==vol)&(data['modality']=='cbv')&(data['duration']==stimDur)]
        # tmp = noOutliers.loc[(noOutliers['volume']==vol)&(noOutliers['modality']=='cbv')&(noOutliers['duration']==stimDur)]
        tmpVals = np.asarray(tmp['val'])
        tmpMean = np.mean(tmpVals)
        cbvMean[i] = tmpMean


    boldMean = np.zeros(len(data['volume'].unique()))
    volumes = np.sort(data['volume'].unique())
    for i, vol in enumerate(volumes):
        tmp = data.loc[(data['volume']==vol)&(data['modality']=='bold')&(data['duration']==stimDur)]
        # tmp = noOutliers.loc[(noOutliers['volume']==vol)&(noOutliers['modality']=='bold')&(noOutliers['duration']==stimDur)]
        tmpVals = np.asarray(tmp['val'])
        tmpMean = np.mean(tmpVals)
        boldMean[i] = tmpMean

    vaso = np.divide(cbvMean,boldMean)

    fig, ax = plt.subplots(1,1,figsize=(7.5,5))
    ax.axvspan(0, stimDur / tr, color='#e5e5e5', alpha=0.2, lw=0)

    plt.title(f'{stimDur}s second stimulation', fontsize=24,pad=10)

    plt.plot(((vaso/-vaso[0])+1)*100, label='vaso', linewidth = 3, color = 'tab:blue')
    plt.plot(boldMean-100, label='bold', linewidth = 3, color= 'tab:orange')

    ax.set_ylabel('Signal change [%]', fontsize=24)
    plt.ylim(-2,4.5)
    # Prepare ticks for x axis
    xticks = np.linspace(0, maxVol, 8)
    xlabels = (xticks * tr).round(decimals = 1)
    ax.set_xticks(xticks, xlabels.astype('int'), fontsize = 18)
    ax.set_xlabel('Time [s]', fontsize=24)
    ax.axhline(0,linestyle = '--', color = 'white')

    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)

    plt.legend()
    plt.legend(title = 'Modality', fontsize=14, title_fontsize=18)
    plt.savefig(f'./results/boldVaso_{stimDur}.png', bbox_inches = "tight")

    plt.show()



# ==============================================================================
# Extract voxel wise ERAs in high temporal resolution
# ==============================================================================
# Set event durations for runs with short and long ITIs
EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

for sub in ['sub-05','sub-07','sub-09']:  # Loop over participants
    for modality in ['bold', 'cbv']:
    # for modality in ['cbv']:

        for k, stimDur in enumerate([1,2,4,12,24]):
            print(f'Processing stimulus duration: {stimDur}s')

            maxVol = np.amax(data.loc[(data['modality']=='cbv') & (data['duration']==stimDur)]['volume'])

            runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-stimulation_run-avg_part-mag_{modality}.nii'))

            # Get dimensions
            nii = nb.load(runs[-1])
            dims = nii.header['dim'][1:-4]
            dims = np.append(dims,maxVol)

            header = nii.header
            affine = nii.affine

            newData = np.zeros(dims)
            divisor = np.zeros(dims)

            for runFile in runs:

                tmpData = np.zeros(dims)
                tmpDivisor = np.zeros(dims)

                base = os.path.basename(runFile).rsplit('.', 2)[0]
                print(f'Processing {base}')
                for i in range(1,99):
                    if f'ses-0{i}' in base:
                        ses = f'ses-0{i}'



                log = f'code/stimulation/{sub}/{ses}/{sub}_{ses}_run-01_neurovascularCoupling.log'
                logFile = pd.read_csv(log,usecols=[0])

                # Because the column definition will get hickups if empty colums are
                # present, we find line with first trigger to then load the file anew,
                # starting with that line
                for index, row in logFile.iterrows():
                    if re.search('Keypress: 5', str(row)):
                        firstVolRow = index
                        break

                # Define column names
                ColNames = ['startTime', 'type', 'event']

                # load logfile again, starting with first trigger
                logFile = pd.read_csv(log, sep = '\t',skiprows=firstVolRow, names = ColNames)

                runNii = nb.load(runFile)
                runData = runNii.get_fdata()

                # See if run has long or sort ITIs based on #volumes
                if runData.shape[-1] > 250:
                    iti = 'longITI'
                if runData.shape[-1] < 250:
                    iti = 'shortITI'

                # Linearly detrend data if wanted
                signal.detrend(runData, axis = -1, type = 'linear', overwrite_data = True)

                # Set baseline for normalization
                baseline1 = np.mean(runData[...,:11], axis = -1)
                baseline2 = np.mean(runData[...,-21:], axis = -1)
                baseline = np.add(baseline1,baseline2) / 2

                # Give baseline the same shape as data
                tmpMean = np.zeros(runData.shape)
                for i in range(tmpMean.shape[-1]):
                    tmpMean[...,i] = baseline

                # Compute signal change with reference to baseline
                sigChange = (np.divide(runData, tmpMean)) * 100
                # sigChange = (np.divide(runData, tmpMean) - 1) * 100

                # Initiate lists
                starts = []
                jitters = []

                stimSwitch = False

                # Loop over lines of log and find stimulation start and stop times
                for index, row in logFile.iterrows():
                    if not logFile['event'][index] != logFile['event'][index]:

                        if re.search('stimDur', logFile['event'][index]):
                            currStimDur = int(float(re.findall(r"\d+\.\d+", logFile['event'][index])[0]))

                            if currStimDur == stimDur:
                                stimSwitch = True

                        if stimSwitch:
                            if re.search('jitter', logFile['event'][index]):
                                currJitter = float(re.findall(r"\d+\.\d+", logFile['event'][index])[0])
                                jitters.append(currJitter)

                            if modality == 'bold':
                                if re.search('TR2', logFile['event'][index]):
                                    if currJitter <= 2:
                                        boldVol = int(re.findall(r"\d+", logFile['event'][index])[-1])  # because of 0-indexing of data
                                        starts.append(boldVol)
                                        stimSwitch = False
                                    if currJitter >= 2:
                                        boldVol = int(re.findall(r"\d+", logFile['event'][index])[-1])+1  # because of 0-indexing of data
                                        starts.append(boldVol)
                                        stimSwitch = False
                            if modality == 'cbv':
                                if re.search('TR1', logFile['event'][index]):
                                    if currJitter <= 0.5:
                                        cbvVol = int(re.findall(r"\d+", logFile['event'][index])[-1])  # because of 0-indexing of data
                                        starts.append(cbvVol)
                                        stimSwitch = False
                                    if currJitter >= 0.5:
                                        cbvVol = int(re.findall(r"\d+", logFile['event'][index])[-1])+1  # because of 0-indexing of data
                                        starts.append(cbvVol)
                                        stimSwitch = False
                if modality == 'bold':
                    jitters = [boldJitters[truncate(i, 3)] for i in jitters]
                if modality == 'cbv':
                    jitters = [cbvJitters[truncate(i, 3)] for i in jitters]

                length = int(np.round(EVENTDURS[iti][k]/(tr*4)))
                for i, start in enumerate(starts):
                    tmp = sigChange[...,start:start+length]

                    newStart = int(np.round(jitters[i]/tr))

                    for j in range(tmp.shape[-1]):
                        test = int(newStart + int(j*4))

                        tmpData[...,test] += tmp[...,j]
                        tmpDivisor[...,test] += np.ones(dims[:-1])

                print('adding run to sub-data')
                newData += tmpData
                divisor += tmpDivisor

            tmp = np.divide(newData, divisor)

            img = nb.Nifti1Image(tmp, header = header, affine = affine)
            nb.save(img, f'{DATADIR}/{sub}/ERAs/{sub}_task-stimulation_run-avg_part-mag_{modality}_highRes_era-{stimDur}s_sigChange.nii.gz')


for stimDur in [1,2,4,12,24]:
    fig, ax = plt.subplots(1,1,figsize=(7.5,5))
    # Load nii
    nii = nb.load(f'{DATADIR}/{sub}/ERAs/{sub}_task-stimulation_run-avg_part-mag_bold_highres_era-{stimDur}s_sigChange.nii.gz')
    # Get data
    niiData = nii.get_fdata()

    # Set and load mask data
    maskFile = f'{ROOT}/{sub}/v1Mask.nii.gz'
    maskData = nb.load(maskFile).get_fdata()

    vals = []
    tps = []


    for vol in range(niiData.shape[-1]):
        val = np.mean(niiData[...,vol][maskData.astype(bool)])
        vals.append(val)
        tps.append(vol)


    sns.lineplot(data = data.loc[(data['run'].str.contains('sub-06'))&(data['modality']=='cbv')&(data['duration']==stimDur)],
                 x = 'volume',
                 y = 'val',
                 linewidth = 3,
                 # color = '#ff8c26'
                 color = 'white'
                 )

    plt.plot(tps,-np.asarray(vals))
    plt.show()


# ==============================================================================
# Extract voxel wise ERAs in high temporal resolution and upsampled space
# ==============================================================================

SUBS = ['sub-05', 'sub-06','sub-07','sub-09']
SUBS = ['sub-07']

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []
depthList = []

for sub in SUBS:
    subDir = f'{DATADIR}/{sub}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    depthFile = glob.glob(f'{segFolder}/3layers_layers_equivol.nii*')[0]
    depthNii = nb.load(depthFile)
    depthData = depthNii.get_fdata()
    layers = np.unique(depthData)[1:]

    # roisData = nb.load(f'{roiFolder}/sub-05_vaso_stimulation_registered_crop_largestCluster_bin_UVD_max_filter.nii.gz').get_fdata()
    roisData = nb.load(glob.glob(f'{segFolder}/{sub}_rim*_perimeter_chunk.nii*')[0]).get_fdata()
    roiIdx = roisData == 1


    for stimDuration in [1, 2, 4, 12, 24]:
    # for stimDuration in [1, 12]:

        for modality in ['cbv', 'bold']:
        # for modality in ['bold']:
            # frames = sorted(glob.glob(f'{DATADIR}/{sub}/ERAs/frames/{sub}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDuration}s_sigChange_masked_frame*_registered_crop.nii.gz'))
            frames = sorted(glob.glob(f'{DATADIR}/{sub}/ERAs/frames/{sub}_task-stimulation_run-avg_part-mag_{modality}_highRes_era-{stimDuration}s_sigChange_frame*_registered_crop.nii.gz'))
            # file = f'{DATADIR}/{sub}/ERAs/frames/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-12s_sigChange-after_frame12_registered_crop.nii.gz'
            frames = frames[:-3]
            for j,frame in enumerate(frames):

                nii = nb.load(frame)

                data = nii.get_fdata()

                for layer in layers:

                    layerIdx = depthData == layer
                    tmp = roiIdx * layerIdx

                    val = np.mean(data[tmp])

                    valList.append(val)
                    subList.append(sub)
                    depthList.append(layer)
                    stimDurList.append(stimDuration)
                    modalityList.append(modality)
                    timePointList.append(j)

data = pd.DataFrame({'subject': subList, 'volume': timePointList, 'modality': modalityList, 'val': valList, 'layer':depthList, 'duration':stimDurList})


palettesLayers = {'vaso':['#55a8e2','#aad4f0','#ffffff','#FF0000'],
'bold':['#ff8c26', '#ffd4af','#ffffff','#FF0000']}
layerNames = ['deep', 'middle', 'superficial','vein']


for k, stimDur in enumerate([1,2,4,12,24]):
    cbvMean = np.zeros(len(data['volume'].unique()))
    volumes = np.sort(data['volume'].unique())
    fig, ax = plt.subplots(1,1,figsize=(7.5,5))


    maxVol = np.amax(data.loc[(data['modality']=='cbv')&(data['duration']==24)]['volume'])

    for z, layer in enumerate(data['layer'].unique()):

        for i, vol in enumerate(volumes):
            tmp = data.loc[(data['volume']==vol)&(data['modality']=='cbv')&(data['duration']==stimDur)&(data['layer']==layer)]
            # tmp = noOutliers.loc[(noOutliers['volume']==vol)&(noOutliers['modality']=='cbv')&(noOutliers['duration']==stimDur)]
            tmpVals = np.asarray(tmp['val'])
            tmpMean = np.mean(tmpVals)
            cbvMean[i] = tmpMean


        boldMean = np.zeros(len(data['volume'].unique()))
        volumes = np.sort(data['volume'].unique())
        for i, vol in enumerate(volumes):
            tmp = data.loc[(data['volume']==vol)&(data['modality']=='bold')&(data['duration']==stimDur)&(data['layer']==layer)]
            # tmp = noOutliers.loc[(noOutliers['volume']==vol)&(noOutliers['modality']=='bold')&(noOutliers['duration']==stimDur)]
            tmpVals = np.asarray(tmp['val'])
            tmpMean = np.mean(tmpVals)
            boldMean[i] = tmpMean

        vaso = np.divide(cbvMean,boldMean)

        plt.plot(((vaso*-1)+1)*100, label=f'{layerNames[z]}', linewidth = 3, color = palettesLayers['vaso'][z])
        # plt.plot(boldMean-100, label=f'{layerNames[z]}', linewidth = 3, color= palettesLayers['bold'][z])

    ax.axvspan(0, stimDur / tr, color='#e5e5e5', alpha=0.2, lw=0)

    plt.title(f'{stimDur}s second stimulation', fontsize=24,pad=10)


    ax.set_ylabel('Signal change [%]', fontsize=24)
    # plt.ylim(-2,4)
    # Prepare ticks for x axis
    # xticks = np.linspace(0, maxVol, 8)
    # xlabels = (xticks * tr).round(decimals = 1)
    xticks = np.linspace(0, maxVol, 8)
    xlabels = (xticks * tr).round(decimals = 1)
    ax.set_xticks(xticks, xlabels.astype('int'), fontsize = 18)
    ax.set_xlabel('Time [s]', fontsize=24)
    ax.axhline(0,linestyle = '--', color = 'white')

    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)

    plt.legend()
    plt.legend(title = 'Modality', fontsize=14, title_fontsize=18)
    plt.show()
