''' Get event-related averages per stimulus duration '''

import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Define root dir
ROOT = '/Users/sebastiandresbach/git/neurovascularCouplingVASO'
# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-01'

# Define subjects to work on
subs = ['sub-01']

def findTR(logfile):
    with open(logfile) as f:
        f = f.readlines()

    triggerTimes = []
    for line in f[1:]:
        if re.findall("Keypress: 5",line):
            triggerTimes.append(float(re.findall("\d+\.\d+", line)[0]))

    triggerTimes[0] = 0

    triggersSubtracted = []
    for n in range(len(triggerTimes)-1):
        triggersSubtracted.append(float(triggerTimes[n+1])-float(triggerTimes[n]))

    meanFirstTriggerDur = np.mean(triggersSubtracted[::2])
    meanSecondTriggerDur = np.mean(triggersSubtracted[1::2])

    # Find mean trigger-time
    meanTriggerDur = (meanFirstTriggerDur+meanSecondTriggerDur)/2
    return meanTriggerDur


# =============================================================================
# Extract upsampled timecourse
# =============================================================================

timePointList = []
modalityList = []
valList = []
stimDurList = []
jitList = []

for sub in subs:
    logFile = f'/Users/sebastiandresbach/git/neurovascularCouplingVASO/code/stimulation/sub-01/ses-01/{sub}_ses-01_run-01_neurovascularCoupling.log'
    tr = findTR(logFile)

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

        startTRs = (trialStart / (tr/3))
        stopTRs = (trialStop / (tr/3))

        for startTR, stopTR in zip(startTRs,stopTRs):
            allStartTRs.append(startTR)
            allStopTRs.append(startTR + (stimDuration / (tr/3)))

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

        startTRs = (trialStart / (tr/3))
        stopTRs = (trialStop / (tr/3))

        nrTRs = int(np.mean(stopTRs-startTRs, axis=0))
        stopTRs = startTRs+nrTRs


        trials = np.zeros((len(startTRs),nrTRs))

        for modality in ['vaso', 'bold']:
        # for modality in ['bold']:

            mriData = np.load(f'{DATADIR}/{sub}_task-stimulation_part-mag_{modality}_intemp_timecourse_intemp.npy')
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

                trials[i,:] = tmp
                if modality == 'vaso':
                    tmp = -tmp
                for j, item in enumerate(tmp):
                    timePointList.append(j)
                    modalityList.append(modality)
                    valList.append(item)
                    stimDurList.append(stimDuration)
                    jitList.append(jitterList[i])


data = pd.DataFrame({'volume': timePointList, 'modality': modalityList, 'data': valList, 'stimDur': stimDurList, 'jitter': jitList})
plt.style.use('dark_background')

palette = {
    'bold': 'tab:orange',
    'vaso': 'tab:blue'}

# =============================================================================
# Plot modality means
# =============================================================================

for stimDuration in [1., 2., 4., 12., 24.]:
    fig, (ax1) = plt.subplots(1,1,figsize=(7.5,5))

    for modality in ['bold', 'vaso']:

        val = np.mean(data.loc[(data['stimDur'] == stimDuration)&(data['volume'] == 0)&(data['modality'] == modality)]['data'])
        tmp = data.loc[(data['stimDur'] == stimDuration)&(data['modality'] == modality)]
        tmp['data'] = tmp['data'] - val

        sns.lineplot(ax=ax1,
                     data = tmp,
                     x="volume",
                     y="data",
                     color=palette[modality],
                     linewidth=3,
                     ci=None,
                     label=modality,
                     )

    # prepare x-ticks
    ticks = range(0,tmp.shape[0]+1,4)
    labels = (np.arange(0,tmp.shape[0]+1,4)*0.5).round(decimals=1).astype('int')

    ax1.set_yticks(np.arange(-0.25,1.26,0.25))

    ax1.yaxis.set_tick_params(labelsize=18)
    ax1.xaxis.set_tick_params(labelsize=18)

    # tweak x-axis
    ax1.set_xticks(ticks[::2])
    ax1.set_xticklabels(labels[::2],fontsize=18)
    ax1.set_xlabel('Time [s]', fontsize=24)

    # draw lines
    ax1.axvspan(0, stimDuration*2, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation')
    # get value of first timepoint

    ax1.axhline(0,linestyle='--',color='white')

    legend = ax1.legend(loc='upper right', title="Modalities", fontsize=14)
    legend.get_title().set_fontsize('16') #legend 'Title' fontsize

    fig.tight_layout()

    ax1.set_ylabel(r'Signal change [%]', fontsize=24)

    if stimDuration == 1:
        plt.title(f'{int(stimDuration)} second stimulation', fontsize=24,pad=10)
    else:
        plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24,pad=10)

    plt.savefig(f'/Users/sebastiandresbach/git/neurovascularCouplingVASO/results/sub-01_{int(stimDuration)}.png', bbox_inches = "tight")

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
        sns.lineplot(ax=axis,
                     data=tmp.loc[tmp['modality'] == modality],
                     x="volume",
                     y="data",
                     hue='jitter',
                     linewidth=2,
                     ci=None,
                     palette=palettes[modality]
                     )

        # prepare x-ticks
        ticks = range(0,tmp.shape[0]+1,4)
        labels = (np.arange(0,tmp.shape[0]+1,4)*0.5).round(decimals=1).astype('int')

        hand, lab = axis.get_legend_handles_labels()
        # sort both labels and handles by labels
        lab, hand = zip(*sorted(zip(lab, hand), key=lambda t: t[0]))

        axis.set_yticks(np.arange(-1,2.1,0.5))

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

    ax1.set_ylabel(r'Signal change [%]', fontsize=24)
    ax2.set_ylabel(r'', fontsize=24)
    plt.savefig(f'../../results/sub-01_{int(stimDuration)}_jitters.png', bbox_inches = "tight")

    plt.show()



# =============================================================================
# Extract timecourse
# =============================================================================


for sub in subs:
    fig = plt.figure()
    logFile = f'../stimulation/{sub}/ses-01/{sub}_ses-01_run-01_neurovascularCoupling.log'
    tr = findTR(logFile)

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


    # for stimDuration in [1., 2., 4., 12., 24.]:
    for stimDuration in [2.]:

        # initiate lists
        trialStart = []
        trialStop = []
        jitterList = []

        stimSwitch = False
        # loop over lines and fine stimulation start and stop times
        for index, row in logData.iterrows():
            if not logData['event'][index] != logData['event'][index]:  # Exclude NaNs
                if re.search(f'stimDur = {stimDuration}', logData['event'][index]):
                    trialStart.append(logData['startTime'][index])
                    stimSwitch = True
                if re.search(f'jitter', logData['event'][index]) and stimSwitch == True:
                    jitterList.append(logData['event'][index][-3:])
                if re.search('Trial complete', logData['event'][index]) and stimSwitch == True:
                    trialStop.append(logData['startTime'][index])
                    stimSwitch = False



        trialStart = np.asarray(trialStart)
        trialStop = np.asarray(trialStop)

        startTRs = (trialStart / tr)
        stopTRs = (trialStop / tr)

        nrTRs = int(np.mean(stopTRs-startTRs, axis=0))
        stopTRs = startTRs+nrTRs


        trials = np.zeros((len(startTRs),nrTRs))


        fig = plt.figure()
        # plt.title(stimDuration)

        # for modality in ['vaso', 'bold']:
        for modality in ['vaso']:

            run = f'{DATADIR}/sub-01_task-stimulation_part-mag_{modality}_intemp.nii.gz'

            nii = nb.load(run)
            header = nii.header
            data = nii.get_fdata()

            mask = nb.load(f'{DATADIR}/v1Mask.nii.gz').get_fdata()

            # Because we want the % signal-change, we need the mean
            # of the voxels we are looking at.
            mask_mean = np.mean(data[:, :, :][mask.astype(bool)])

            for i, (start, end) in enumerate(zip(startTRs, stopTRs)):
                tmp = np.mean((((data[:, :, :, int(start):int(end)][mask.astype(bool)]) / mask_mean)- 1) * 100,axis=0)
                # print(tmp.shape)
                trials[i,:] = tmp
                plt.plot(tmp, linewidth=2, label = f'{stimDuration}s jitter: {jitterList[i]}')

            if modality == 'vaso':
                trials = -trials
            avg = np.mean(trials,axis=0)

        plt.legend()
        plt.show()


    plt.show()





startTRs
stopTRs

for sub in ['sub-01']:
    for modality in ['cbv', 'bold']:

        outFolder = f'{ROOT}/derivatives/{sub}/'

        # find all runs of participant
        allRuns = sorted(glob.glob(f'{ROOT}/derivatives/{sub}/ses-0*/{sub}_ses-0*_task-stimulation_run-0*_part-mag_{modality}_moco_registered.nii'))
        firstRun = sorted(glob.glob(f'{ROOT}/derivatives/{sub}/ses-01/{sub}_ses-01_task-stimulation_run-01_part-mag_{modality}_moco.nii'))
        allRuns.insert(0, firstRun[0])
        nrRuns = len(allRuns)

        # find highest number of volumes
        highstVolNr = 0

        for run in allRuns:
            nii = nb.load(run)
            header = nii.header
            dataShape = header.get_data_shape()
            nrVolumes = dataShape[-1]
            if nrVolumes > highstVolNr:
                highstVolNr = nrVolumes

        newShape = (
            dataShape[0],
            dataShape[1],
            dataShape[2],
            highstVolNr
            )
        newData = np.zeros(newShape)
        divisor = np.zeros(newShape)

        for run in allRuns:
            nii = nb.load(run)
            header = nii.header
            data = nii.get_fdata()
            nrVolumes = data.shape[-1]


            newData[:,:,:,:nrVolumes] += data
            divisor[:,:,:,:nrVolumes] += 1

        newData = newData/divisor

        nii = nb.load(allRuns[0])
        header = nii.header
        affine = nii.affine

        # save image
        img = nb.Nifti1Image(newData, header=header, affine=affine)
        nb.save(img, f'{outFolder}/{sub}_task-stimulation_part-mag_{modality}.nii')


modalities = glob.glob(f'{outFolder}/{sub}_task-stimulation_part-mag_*.nii')
# combining cbv and bold weighted images
os.system(f'{afniPath}/3dTcat -prefix {outFolder}/{sub}_task-stimulation_part-mag_combined.nii  {modalities[0]} {modalities[1]} -overwrite')
# Calculating T1w image in EPI space for each run
os.system(f'{afniPath}/3dTstat -cvarinv -overwrite -prefix {outFolder}/{sub}_task-stimulation_part-mag_T1w.nii {outFolder}/{sub}_task-stimulation_part-mag_combined.nii')
# Running biasfieldcorrection
os.system(f'{antsPath}/N4BiasFieldCorrection -d 3 -i {outFolder}/{sub}_task-stimulation_part-mag_T1w.nii -o {outFolder}/{sub}_task-stimulation_part-mag_T1w_N4Corrected.nii')
