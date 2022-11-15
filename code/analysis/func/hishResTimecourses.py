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

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

maskFile = f'{ROOT}/sub-06/v1Mask.nii.gz'
maskData = nb.load(maskFile).get_fdata()


runFile = f'{ROOT}/sub-06/ses-03/func/sub-06_ses-03_task-stimulation_run-avg_part-mag_cbv.nii'
runNii = nb.load(runFile)
runData = runNii.get_fdata()

timecourse = np.mean(runData[:, :, :][maskData.astype(bool)], axis=0)
mean = np.mean(timecourse, axis = -1)
sigChange = (np.divide(timecourse, mean) - 1) * 100


starts = [104, 118, 179, 193]
jitters = np.asarray([0.785, 0, 1.57, 2.355])

boldJitters = {0.785: 1.57, 0:2.355, 1.57:0, 2.355:0.785}

boldJitters[1.57]

jitters = [boldJitters[i] for i in jitters]

jitters


jitters -= 1.57

jitters = np.absolute(jitters)

jitters[1] = 2.355
jitters[0] = 1.57

tr = 0.785

length = 15

new = np.zeros(length*len(jitters))
plt.figure()

timepointList = []
runList = []
valList = []

test = np.arange(0,4)
orders = list(permutations(test, 4))

for order in orders:
    print(order)
    timepointList = []
    runList = []
    valList = []
    for ses in ['ses-03','ses-05']:
        runs = sorted(glob.glob(f'{ROOT}/sub-06/{ses}/func/sub-06_{ses}_task-stimulation_run-0*_part-mag_bold_moco-reg.nii'))

        for runFile in runs:
            base = os.path.basename(runFile).rsplit('.', 2)[0]

            runNii = nb.load(runFile)
            runData = runNii.get_fdata()

            timecourse = np.mean(runData[:, :, :][maskData.astype(bool)], axis=0)
            mean = np.mean(timecourse, axis = -1)
            sigChange = (np.divide(timecourse, mean) - 1) * 100


            for i, start in enumerate(starts):

                tmp = sigChange[start:start+length]
                tmp = tmp - np.mean(tmp)

                newStart = order[i]

                for j, item in enumerate(tmp):

                    test = int(newStart + int(j*4))

                    timepointList.append(test)
                    valList.append(item)
                    runList.append(base)

                # plt.plot(tmp,label=jitters[i])


    data = pd.DataFrame({'run':runList, 'val': valList, 'volume':timepointList})
    volumes = np.sort(data['volume'].unique())

    for vol in volumes:
        tmp = data.loc[data['volume']==vol]
        tmpVals = np.asarray(tmp['val'])

        percentile25 = tmp['val'].quantile(0.25)
        percentile75 = tmp['val'].quantile(0.75)

        iqr = percentile75-percentile25

        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr

        for val in tmpVals:
            if val < lower_limit:
                data.drop(data.loc[data['val']==val].index, inplace=True)

        for val in tmpVals:
            if val > upper_limit:
                data.drop(data.loc[data['val']==val].index, inplace=True)


    plt.figure()
    sns.scatterplot(data=data, x="volume", y="val", hue='run')
    sns.lineplot(data=data, x='volume', y='val', linewidth=3, color='black')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()






#####

# saving original here



# for ses in ['ses-01','ses-02','ses-03','ses-05']:
# for ses in ['ses-03','ses-05']:
subs = ['sub-05','sub-06','sub-07','sub-08']
subs = ['sub-06']
boldJitters = {0.785: 1.570, 0.0: 2.355, 1.57: 0, 2.355: 0.785}
# sort in oppisite order

boldJitters = {0.0: 2.355, 0.785: 1.570, 1.57: 0, 2.355: 0.785}

boldJitters = {0.0: 2.355, 0.785: 1.570, 1.57: 0.785, 2.355: 0}

# boldJitters = {0.0: 2.355, 0.785: 1.570, 1.57: 0, 2.355: 0.785}


jitters = np.asarray(jitters)
starts = np.asarray(starts)

sort = np.argsort(jitters)
print(jitters[sort])
print(jitters[~sort])
print(starts[sort])

tr = 0.785
EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

28/(tr*2)

timepointList = []
runList = []
valList = []
stimDurList = []
modalityList = []
def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

for sub in subs:
    for modality in ['bold', 'cbv']:
        # runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-stimulation_run-0*_part-mag_bold_moco-reg.nii'))
        runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-stimulation_run-0*_part-mag_{modality}_moco-reg.nii'))

        for runFile in runs:
            base = os.path.basename(runFile).rsplit('.', 2)[0]
            print(f'Processing {base}')
            for i in range(1,99):
                if f'ses-0{i}' in base:
                    ses = f'ses-0{i}'
            for i in range(1,99):
                if f'run-0{i}' in base:
                    runNr = f'run-0{i}'

            log = f'code/stimulation/{sub}/{ses}/{sub}_{ses}_{runNr}_neurovascularCoupling.log'
            logFile = pd.read_csv(log,usecols=[0])

            # Because the column definition will get hickups if empty colums are
            # present, we find line with first trigger to then load the file anew,
            # starting with that line
            for index, row in logFile.iterrows():
                if re.search('Keypress: 5', str(row)):
                    firstVolRow = index
                    break

            # define column names
            ColNames = ['startTime', 'type', 'event']
            # load logfile again, starting with first trigger
            # logFile = pd.read_csv(f'{ROOT}/derivatives/{sub}/{ses}/events/{base}.log', sep = '\t',skiprows=firstVolRow, names = ColNames)

            logFile = pd.read_csv(log, sep = '\t',skiprows=firstVolRow, names = ColNames)

            for k, stimDur in enumerate([1,2,4,12,24]):
                # initiate lists
                starts = []
                jitters = []

                stimSwitch = False

                # loop over lines and fine stimulation start and stop times
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
                            if re.search('TR1', logFile['event'][index]):
                                boldVol = int(re.findall(r"\d+", logFile['event'][index])[-1])-1  # because of 0-indexing of data
                                starts.append(boldVol)
                                stimSwitch = False

                if modality == 'bold':
                    jitters = [boldJitters[truncate(i, 3)] for i in jitters]

                runNii = nb.load(runFile)
                runData = runNii.get_fdata()

                if runData.shape[-1] > 250:
                    iti = 'longITI'
                if runData.shape[-1] < 250:
                    iti = 'shortITI'
                    k = 0

                length = int(np.round(EVENTDURS[iti][k]/(tr*2)))

                signal.detrend(runData, axis = -1, type = 'linear', overwrite_data = True)

                timecourse = np.mean(runData[:, :, :][maskData.astype(bool)], axis=0)

                # timecourse_de = signal.detrend(timecourse)

                mean = np.mean(timecourse, axis = -1)
                sigChange = (np.divide(timecourse, mean) - 1) * 100

                for i, start in enumerate(starts):

                    tmp = sigChange[start:start+length]

                    tmp = tmp - np.mean(tmp)

                    newStart = int(np.round(jitters[i]/tr))

                    for j, item in enumerate(tmp):
                        test = int(newStart + int(j*4))

                        timepointList.append(test)
                        valList.append(item)
                        runList.append(base)
                        stimDurList.append(stimDur)
                        modalityList.append(modality)

            # plt.plot(tmp,label=jitters[i])


data = pd.DataFrame({'run':runList, 'val': valList, 'volume':timepointList, 'duration': stimDurList, 'modality': modalityList})

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


        # plt.figure()
        # sns.histplot(data=tmp, x="val",hue='run', binwidth=0.5, binrange = [-10, 8])
        # sns.kdeplot(data=tmp, x="val")
        # sns.rugplot(data=tmp, x="val")
        # sns.boxplot(data=tmp, y="val")

        # plt.title(vol)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.show()

for modality in ['bold', 'cbv']:
    for k, stimDur in enumerate([1,2,4,12,24]):
        tmp = noOutliers.loc[(noOutliers['duration']==stimDur)&(noOutliers['modality']==modality)]
        # tmp = data.loc[(data['duration']==stimDur)]
        fig, ax = plt.subplots()
        sns.scatterplot(data=tmp, x="volume", y="val", hue='run', legend=False)
        sns.lineplot(data=tmp, x='volume', y='val', linewidth=3, color='black')
        plt.xlim(0,100)
        plt.ylim(-4,7)
        ax.set_ylabel('Signal change [%]')
        ax.set_xlabel('Volume')
        plt.title(f'{modality} {stimDur}s')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f'./results/{modality}_{stimDur}_newWorldOrder1.png', bbox_inches = "tight")
        plt.show()


cbvMean = np.zeros(len(data['volume'].unique()))
volumes = np.sort(data['volume'].unique())

for i, vol in enumerate(volumes):
    tmp = data.loc[(data['volume']==vol)&(data['modality']=='cbv')&(data['duration']==24)]
    tmpVals = np.asarray(tmp['val'])
    tmpMean = np.mean(tmpVals)
    cbvMean[i] = tmpMean

plt.plot(cbvMean)



boldMean = np.zeros(len(data['volume'].unique()))
volumes = np.sort(data['volume'].unique())
for i, vol in enumerate(volumes):
    tmp = data.loc[(data['volume']==vol)&(data['modality']=='bold')&(data['duration']==24)]
    tmpVals = np.asarray(tmp['val'])
    tmpMean = np.mean(tmpVals)
    boldMean[i] = tmpMean



vaso = np.divide(cbvMean,boldMean)

plt.plot(-vaso, label='vaso')
# plt.plot(boldMeanSmooth, label='bold')
# plt.plot(cbvMeanSmooth, label='nulled')
plt.ylim(-3,3)
plt.legend()
