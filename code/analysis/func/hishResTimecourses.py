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
# subs = ['sub-06']

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

    for modality in ['bold', 'cbv']:
    # for modality in ['bold']:

        runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-stimulation_run-avg_part-mag_{modality}.nii'))

        for runFile in runs:
            base = os.path.basename(runFile).rsplit('.', 2)[0]
            print(f'Processing {base}')
            for i in range(1,99):
                if f'ses-0{i}' in base:
                    ses = f'ses-0{i}'
            for i in range(1,99):
                if f'run-0{i}' in base:
                    runNr = f'run-0{i}'

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
                                        cbvVol = int(re.findall(r"\d+", logFile['event'][index])[-1])-1  # because of 0-indexing of data
                                        starts.append(cbvVol)
                                        stimSwitch = False
                                    if currJitter >= 0.5:
                                        cbvVol = int(re.findall(r"\d+", logFile['event'][index])[-1])  # because of 0-indexing of data
                                        starts.append(cbvVol)
                                        stimSwitch = False
                if modality == 'bold':
                    jitters = [boldJitters[truncate(i, 3)] for i in jitters]
                if modality == 'cbv':
                    jitters = [cbvJitters[truncate(i, 3)] for i in jitters]

                length = int(np.round(EVENTDURS[iti][k]/(tr*4)))+1

                # print(f'stimdur: {stimDur}, ITI: {iti}, extracting {length} timpoints')

                for i, start in enumerate(starts):

                    tmp = sigChange[start:start+length]
                    # tmp = timecourse[start:start+length]

                    # tmp = tmp - np.mean(tmp)

                    newStart = int(np.round(jitters[i]/tr))

                    for j, item in enumerate(tmp):
                        test = int(newStart + int(j*4))

                        timepointList.append(test)
                        valList.append(item)
                        runList.append(base)
                        stimDurList.append(stimDur)
                        modalityList.append(modality)



data = pd.DataFrame({'run':runList, 'val': valList, 'volume':timepointList, 'duration': stimDurList, 'modality': modalityList})

data.to_csv('./results/highresTimecourse.csv', index = False)


noOutliers = data

volumes = np.sort(data['volume'].unique())
for modality in ['bold', 'cbv']:
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
                if val < lower_limit:
                    noOutliers.drop(noOutliers.loc[noOutliers['val']==val].index, inplace=True)

            for val in tmpVals:
                if val > upper_limit:
                    noOutliers.drop(noOutliers.loc[data['val']==val].index, inplace=True)


plt.style.use('dark_background')

for modality in ['bold', 'cbv']:
# for modality in ['bold']:
    for k, stimDur in enumerate([1,2,4,12,24]):
        tmp = noOutliers.loc[(noOutliers['duration']==stimDur)&(noOutliers['modality']==modality)]
        # tmp = data.loc[(data['duration']==stimDur)&(data['modality']==modality)]
        maxVol = np.amax(data['volume'])
        fig, ax = plt.subplots(1,1,figsize=(7.5,5))

        ax.axvspan(0, stimDur / tr, color='#e5e5e5', alpha=0.2, lw=0)


        sns.scatterplot(data=tmp, x="volume", y="val", hue='run', legend=False)
        # Plot mean
        sns.lineplot(data = tmp,
                     x = 'volume',
                     y = 'val',
                     linewidth = 3,
                     # color = '#ff8c26'
                     color = 'black'
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
        # plt.savefig(f'./results/{modality}_{stimDur}.png', bbox_inches = "tight")
        plt.show()



for k, stimDur in enumerate([1,2,4,12,24]):
    cbvMean = np.zeros(len(data['volume'].unique()))
    volumes = np.sort(data['volume'].unique())


    maxVol = np.amax(data.loc[(data['modality']=='cbv')&(data['duration']==24)]['volume'])

    for i, vol in enumerate(volumes):
        # tmp = data.loc[(data['volume']==vol)&(data['modality']=='cbv')&(data['duration']==stimDur)]
        tmp = noOutliers.loc[(noOutliers['volume']==vol)&(noOutliers['modality']=='cbv')&(noOutliers['duration']==stimDur)]
        tmpVals = np.asarray(tmp['val'])
        tmpMean = np.mean(tmpVals)
        cbvMean[i] = tmpMean


    boldMean = np.zeros(len(data['volume'].unique()))
    volumes = np.sort(data['volume'].unique())
    for i, vol in enumerate(volumes):
        # tmp = data.loc[(data['volume']==vol)&(data['modality']=='bold')&(data['duration']==stimDur)]
        tmp = noOutliers.loc[(noOutliers['volume']==vol)&(noOutliers['modality']=='bold')&(noOutliers['duration']==stimDur)]
        tmpVals = np.asarray(tmp['val'])
        tmpMean = np.mean(tmpVals)
        boldMean[i] = tmpMean

    vaso = np.divide(cbvMean,boldMean)

    fig, ax = plt.subplots(1,1,figsize=(7.5,5))
    ax.axvspan(0, stimDur / tr, color='#e5e5e5', alpha=0.2, lw=0)

    plt.title(f'{stimDur}s second stimulation', fontsize=24,pad=10)

    plt.plot(((vaso*-1)+1)*100, label='vaso', linewidth = 3, color = 'tab:blue')
    plt.plot(boldMean-100, label='bold', linewidth = 3, color= 'tab:orange')

    ax.set_ylabel('Signal change [%]', fontsize=24)
    plt.ylim(-2,4)
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


