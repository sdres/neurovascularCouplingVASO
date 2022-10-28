''' Plotting event-related averages per stimulus duration '''

import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

import sys
# Define current dir
ROOT = os.getcwd()

sys.path.append('./code/misc')
from findTr import *

# Define data dir
DATADIR = f'/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# Define subjects to work on
subs = ['sub-05', 'sub-06','sub-07','sub-08']
# subs = ['sub-08']
# sessions = ['ses-01']

# =============================================================================
# Extract upsampled timecourse
# =============================================================================

timePointList = []
modalityList = []
valList = []
stimDurList = []
jitList = []
interpList = []
subList = []
sesList = []

for sub in subs:
    print(f'Processing {sub}')
    subDir = f'{DATADIR}/{sub}'

    # =========================================================================
    # Look for sessions
    # Collectall runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{DATADIR}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))

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
    # sessions = ['ses-03']
    for ses in sessions:

        sesDir = f'{subDir}/{ses}/func'

        logFile = f'code/stimulation/{sub}/{ses}/{sub}_{ses}_run-01_neurovascularCoupling.log'

        trEff = findTR(logFile)
        trNom = trEff/4

        # get basename of current run
        base = os.path.basename(logFile).rsplit('.', 2)[0]
        # get logfile
        data = pd.read_csv(f'{logFile}'
                                ,usecols=[0])

        # Because the column definition will get hickups if empty colums are
        # present, we find line with first trigger to then load the file anew,
        # starting with that line
        for index, row in data.iterrows():
            if re.search('StartOfRun', str(row)):
                firstVolRow = index+1
                break
        # define column names
        ColNames = ['startTime', 'type', 'event']

        # load logfile again, starting with first trigger
        logData = pd.read_csv(f'{logFile}', sep = '\t', skiprows=firstVolRow, names = ColNames)


        # =============================================================================
        # Get rest timepoints
        # =============================================================================

        allStartTRs = []
        allStopTRs = []

        for stimDuration in [1., 2., 4., 12., 24.]:

            # initiate lists
            trialStart = []
            trialStop = []
            jitterList = []

            stimSwitch = False

            # loop over lines and fine stimulation start and stop times
            for index, row in logData.iterrows():
                if not logData['event'][index] != logData['event'][index]:  # Exclude NaNs

                    if re.search(f'stimDur = {stimDuration}', logData['event'][index]):
                        stimSwitch = True
                    if re.search(f'stimulation started', logData['event'][index]) and stimSwitch == True:
                        trialStart.append(logData['startTime'][index])
                    if re.search(f'jitter', logData['event'][index]) and stimSwitch == True:
                        jitterList.append(logData['event'][index][-3:])
                    if re.search('Trial complete', logData['event'][index]) and stimSwitch == True:
                        trialStop.append(logData['startTime'][index])
                        stimSwitch = False

            trialStart = np.asarray(trialStart)
            trialStop = np.asarray(trialStop)

            startTRs = trialStart / trNom
            stopTRs = trialStop / trNom

            for startTR, stopTR in zip(startTRs,stopTRs):
                allStartTRs.append(startTR)
                allStopTRs.append(startTR + (stimDuration / trNom))

        allStartTRs.sort()
        allStopTRs.sort()

        allStartTRs = np.asarray(allStartTRs).astype('int')
        allStopTRs = np.asarray(allStopTRs).astype('int')


        for stimDuration in [1., 2., 4., 12., 24.]:

            # initiate lists
            trialStart = []
            trialStop = []
            jitterList = []

            stimSwitch = False

            # loop over lines and fine stimulation start and stop times
            for index, row in logData.iterrows():
                if not logData['event'][index] != logData['event'][index]:  # Exclude NaNs

                    if re.search(f'stimDur = {stimDuration}', logData['event'][index]):
                        stimSwitch = True
                    if re.search(f'stimulation started', logData['event'][index]) and stimSwitch == True:
                        trialStart.append(logData['startTime'][index])
                    if re.search(f'jitter', logData['event'][index]) and stimSwitch == True:
                        jitterList.append(logData['event'][index][-3:])
                    if re.search('Trial complete', logData['event'][index]) and stimSwitch == True:
                        trialStop.append(logData['startTime'][index])
                        stimSwitch = False

            trialStart = np.asarray(trialStart)
            trialStop = np.asarray(trialStop)

            startTRs = (trialStart / (trNom))
            stopTRs = (trialStop / (trNom))

            nrTRs = int(np.mean(stopTRs-startTRs, axis=0))
            stopTRs = startTRs+nrTRs


            trials = np.zeros((len(startTRs),nrTRs))

            for modality in ['vaso', 'bold']:
            # for modality in ['bold']:
                mriData = np.load(f'{sesDir}/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp_timecourse.npy')
                # Because we want the % signal-change, we need the mean
                # of the voxels we are looking at.
                mask_mean = np.mean(mriData)

                # # Or we can normalize to the rest priods only
                # restTRs = np.ones(mriData.shape, dtype=bool)
                #
                # for startTR, stopTR in zip(allStartTRs,allStopTRs):
                #     restTRs[startTR:stopTR] = False
                #
                # mask_mean = np.mean(mriData[restTRs])

                for i, (start, end) in enumerate(zip(startTRs, stopTRs)):
                    tmp = ((( mriData[int(start):int(end)] / mask_mean) - 1) * 100)
                    # tmp = ((( mriData[int(start):int(end)] / mask_mean)) * 100)

                    trials[i,:] = tmp
                    if modality == 'vaso':
                        tmp = -tmp
                    for j, item in enumerate(tmp):
                        timePointList.append(j)
                        modalityList.append(modality)
                        valList.append(item)
                        stimDurList.append(stimDuration)
                        jitList.append(jitterList[i])
                        # interpList.append(interpolationType)
                        subList.append(sub)
                        sesList.append(ses)


data = pd.DataFrame({'subject': subList, 'session': sesList, 'volume': timePointList, 'modality': modalityList, 'data': valList, 'stimDur': stimDurList, 'jitter': jitList})

data.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/{sub}_task-stimulation_responses.csv', sep = ',', index=False)

plt.style.use('dark_background')

palette = {
    'bold': 'tab:orange',
    'vaso': 'tab:blue'}


# =============================================================================
# Plot modality means
# =============================================================================

# for interpolationType in ['linear', 'cubic']:
for interpolationType in ['linear']:
    data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/{sub}_task-stimulation_responses.csv', sep = ',')
    # data = data.loc[data['interpolation']==interpolationType]

    for stimDuration in [1., 2., 4., 12., 24.]:
        fig, (ax1) = plt.subplots(1,1,figsize=(7.5,5))

        for modality in ['bold', 'vaso']:

            val = np.mean(data.loc[(data['stimDur'] == stimDuration)
                                 & (data['volume'] == 0)
                                 & (data['modality'] == modality)]['data']
                                 )

            tmp = data.loc[(data['stimDur'] == stimDuration)&(data['modality'] == modality)]

            # if val > 0:
            #     tmp['data'] = tmp['data'] - val
            # if val < 0:
                # tmp['data'] = tmp['data'] + val
            tmp['data'] = tmp['data'] - val
            nrVols = len(np.unique(tmp['volume']))

            # ax1.set_xticks(np.arange(-1.5,3.6))
            ax1.set_ylim(-1.5,3.5)

            sns.lineplot(ax=ax1,
                         data = tmp,
                         x = "volume",
                         y = "data",
                         color = palette[modality],
                         linewidth = 3,
                         # ci=None,
                         label = modality,
                         )

        # Prepare x-ticks
        ticks = np.linspace(0, nrVols, 10)
        labels = (np.linspace(0, nrVols, 10)*trNom).round(decimals=1)

        # ax1.set_yticks(np.arange(-0.25, 3.51, 0.5))

        ax1.yaxis.set_tick_params(labelsize=18)
        ax1.xaxis.set_tick_params(labelsize=18)

        # tweak x-axis
        ax1.set_xticks(ticks[::2])
        ax1.set_xticklabels(labels[::2],fontsize=18)
        ax1.set_xlabel('Time [s]', fontsize=24)

        # draw lines
        ax1.axvspan(0, stimDuration / trNom, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation')
        # get value of first timepoint

        ax1.axhline(0,linestyle = '--', color = 'white')

        legend = ax1.legend(loc='upper right', title="Modalities", fontsize=14)
        legend.get_title().set_fontsize('16') #legend 'Title' fontsize

        fig.tight_layout()

        ax1.set_ylabel(r'Signal change [%]', fontsize=24)

        if stimDuration == 1:
            plt.title(f'{int(stimDuration)} second stimulation', fontsize=24,pad=10)
        else:
            plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24,pad=10)

        # plt.savefig(f'./results/allSubs_stimDur-{int(stimDuration)}.png', bbox_inches = "tight")

        plt.show()

# =============================================================================
# Plot individual jitters
# =============================================================================


palettes = {
    'bold': 'Oranges',
    'vaso': 'Blues'}

data=data.sort_values(by=['jitter'])


for stimDuration in [1., 2., 4., 12., 24.]:
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

    tmp = data.loc[data['stimDur'] == stimDuration]

    for i, (axis, modality) in enumerate(zip([ax1,ax2], ['vaso', 'bold'])):
        val = np.mean(data.loc[(data['stimDur'] == stimDuration)&(data['volume'] == 0)&(data['modality'] == modality)]['data'])
        tmp = data.loc[(data['stimDur'] == stimDuration)&(data['modality'] == modality)]
        tmp['data'] = tmp['data'] - val
        nrVols = len(np.unique(tmp['volume']))

        sns.lineplot(ax=axis,
                     data=tmp.loc[tmp['modality'] == modality],
                     x="volume",
                     y="data",
                     hue='jitter',
                     linewidth=2,
                     # ci=None,
                     palette=palettes[modality]
                     )

        # prepare x-ticks
        ticks = np.linspace(0, nrVols, 10)
        labels = (np.linspace(0, nrVols, 10)*trNom).round(decimals=1)

        hand, lab = axis.get_legend_handles_labels()
        # sort both labels and handles by labels
        lab, hand = zip(*sorted(zip(lab, hand), key=lambda t: t[0]))

        # axis.set_yticks(np.arange(-1,2.1,0.5))

        axis.yaxis.set_tick_params(labelsize=18)
        axis.xaxis.set_tick_params(labelsize=18)

        # tweak x-axis
        axis.set_xticks(ticks[::2])
        axis.set_xticklabels(labels[::2],fontsize=18)
        axis.set_xlabel('Time [s]', fontsize=24)

        # draw lines
        axis.axvspan(0, stimDuration*2, color='#e5e5e5', alpha=0.1, lw=0, label = 'stimulation on')
        axis.axhline(0,linestyle='--',color='white')

        axis.legend(hand, lab,loc='upper right', title="jitters [s]", ncol=2)

    fig.tight_layout()
    if stimDuration == 1:
        plt.suptitle(f'{int(stimDuration)} second stimulation', fontsize=24, y=1.05)
    else:
        plt.suptitle(f'{int(stimDuration)} seconds stimulation', fontsize=24, y=1.05)

    ax1.set_ylabel(r'Signal change [%]', fontsize=24)
    ax2.set_ylabel(r'', fontsize=24)
    # plt.savefig(f'results/{sub}_{int(stimDuration)}_jitters_intemp-linear.png', bbox_inches = "tight")

    plt.show()
