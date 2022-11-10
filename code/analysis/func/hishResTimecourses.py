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

# for ses in ['ses-01','ses-02','ses-03','ses-05']:
for ses in ['ses-03','ses-05']:
    runs = sorted(glob.glob(f'{ROOT}/sub-06/{ses}/func/sub-06_{ses}_task-stimulation_run-0*_part-mag_bold_moco-reg.nii'))

    # try:
    #     runs.remove('/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-03/func/sub-06_ses-03_task-stimulation_run-02_part-mag_bold_moco-reg.nii')
    # except:
    #     print('run not in session')

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

            newStart = int(np.round(jitters[i]/tr))

            for j, item in enumerate(tmp):
                test = int(newStart + int(j*4))

                timepointList.append(test)
                valList.append(item)
                runList.append(base)

            # plt.plot(tmp,label=jitters[i])


data = pd.DataFrame({'run':runList, 'val': valList, 'volume':timepointList})

plt.figure()
sns.scatterplot(data=data, x="volume", y="val", hue='run')
sns.lineplot(data=data, x='volume', y='val', linewidth=2)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()



volumes = np.sort(data['volume'].unique())

for vol in volumes:
    tmp = data.loc[data['volume']==vol]
    tmpVals = np.asarray(tmp['val'])

    # mean = np.mean(tmpVals)
    # stdDev = np.std(tmpVals)
    #
    # if any(i > (mean+(stdDev*2)) for i in tmpVals):
    #     print(f'found outlier in {vol}')
    #
    # if any(i > (mean+(stdDev*2)) for i in tmpVals):
    #     print(f'found outlier in {vol}')
    #
    #
    percentile25 = tmp['val'].quantile(0.25)
    percentile75 = tmp['val'].quantile(0.75)

    iqr = percentile75-percentile25

    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr

    for val in tmpVals:
        if val < lower_limit:
        # print(f'found negative outlier in {vol}')
            data.drop(data.loc[data['val']==val].index, inplace=True)

    for val in tmpVals:
        if val > upper_limit:
        # print(f'found negative outlier in {vol}')
            data.drop(data.loc[data['val']==val].index, inplace=True)
    # if any(i > upper_limit for i in tmpVals):
    #     print(f'found positive outlier in {vol}')

    plt.figure()
    # sns.histplot(data=tmp, x="val",hue='run', binwidth=0.5, binrange = [-10, 8])
    # sns.kdeplot(data=tmp, x="val")
    # sns.rugplot(data=tmp, x="val")
    sns.boxplot(data=tmp, y="val")

    plt.title(vol)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()



plt.show()

# newMean = np.mean(new,axis=0)
# plt.plot(newMean)
# plt.plot(new)
plt.plot(new)
plt.plot(new)
