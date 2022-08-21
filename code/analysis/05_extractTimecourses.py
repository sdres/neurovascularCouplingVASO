''' Get event-related avergaes per stimulus duration '''

import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import pandas as pd
import re
import matplotlib.pyplot as plt
# define root dir
ROOT = '/Users/sebastiandresbach/git/neurovascularCouplingVASO'
# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-01'
# define subjects to work on
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

    # find mean trigger-time
    meanTriggerDur = (meanFirstTriggerDur+meanSecondTriggerDur)/2
    return meanTriggerDur


logFile = f'../stimulation/{sub}/ses-01/{sub}_ses-01_run-01_neurovascularCoupling.log'
tr = findTR(logfile)



for sub in subs:
    fig = plt.figure()

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
    for stimDuration in [1.]:

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
