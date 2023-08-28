"""Compare the superficial layer bias between signal change and zscore"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalized.csv', sep=',')
layerNames = ['deep', 'middle', 'superficial', 'vein']

layerList = []
valList = []
stimDurList = []
dataTypeList = []
timePointList = []
subList = []

for sub in data['subject'].unique():
    for modality in ['vaso']:
        for stimDur in data['stimDur'].unique():
            for layer in data['layer'].unique():
                for dataType in data['dataType'].unique():

                    tmp = data.loc[(data['dataType'] == dataType)
                                   & (data['subject'] == sub)
                                   & (data['modality'] == modality)
                                   & (data['layer'] == layer)
                                   & (data['stimDur'] == stimDur)]

                    maxTP = 0
                    maxVal = 0

                    # Find timepoint with highest value
                    for timePoint in tmp['volume'].unique():
                        val = np.mean(tmp.loc[tmp['volume'] == timePoint]['data'])
                        if val >= maxVal:
                            maxTP = timePoint
                            maxVal = val

                    layerList.append(layerNames[int(layer-1)])
                    valList.append(maxVal)
                    stimDurList.append(stimDur)
                    dataTypeList.append(dataType)
                    timePointList.append(maxTP)
                    subList.append(sub)

peakTimeList = [i * 0.785 for i in timePointList]

timepointData = pd.DataFrame({'subject': subList,
                     'maxVol': timePointList,
                     'data': valList,
                     'layer': layerList,
                     'stimDur': stimDurList,
                     'dataType': dataTypeList,
                     'peakTime': peakTimeList})

timepointData.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/timeToPeak.csv',
            sep=',',
            index=False)

timepointData = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/timeToPeak.csv', sep=',')

palettesLayers = {'vaso': ['#55a8e2', '#aad4f0', '#ffffff', '#FF0000'],
                  'bold': ['#ff8c26', '#ffd4af', '#ffffff', '#FF0000']}

plt.style.use('dark_background')

fig, axes = plt.subplots(1, 5, figsize=(21, 5), sharey=True)
for i, stimDur in enumerate(timepointData['stimDur'].unique()):

    tmp = timepointData.loc[(timepointData['dataType'] == 'raw')
                            & (timepointData['stimDur'] == stimDur)]

    sns.barplot(ax=axes[i], data=tmp, x="stimDur", y="peakTime", hue="layer", palette=palettesLayers['vaso'])

    # find min value
    # minVal = (np.min(tmp['peakTime']) - 1).astype('int')
    # maxVal = (np.max(tmp['peakTime'])).astype('int')
    #
    # axes[i].set_ylim(minVal, maxVal)
    # Prepare legend
    if stimDur == 1:
        legend = axes[i].legend(loc='upper right', title="Layer", fontsize=16)
        legend.get_title().set_fontsize('16')  # Legend 'Title' font-size
    else:
        axes[i].get_legend().remove()
plt.show()

valList = []
stimDurList = []
dataTypeList = []
subList = []

for sub in timepointData['subject'].unique():
    for stimDur in timepointData['stimDur'].unique():
        for dataType in timepointData['dataType'].unique():

            sup = timepointData.loc[(timepointData['dataType'] == dataType)
                               & (timepointData['subject'] == sub)
                               & (timepointData['layer'] == 3.0)
                               & (timepointData['stimDur'] == stimDur)]

            mid = timepointData.loc[(timepointData['dataType'] == dataType)
                               & (timepointData['subject'] == sub)
                               & (timepointData['layer'] == 2.0)
                               & (timepointData['stimDur'] == stimDur)]

            ratio = sup['data'].iloc[0]/mid['data'].iloc[0]

            valList.append(ratio)
            stimDurList.append(stimDur)
            dataTypeList.append(dataType)
            subList.append(sub)

ratioData = pd.DataFrame({'subject': subList,
                     'data': valList,
                     'stimDur': stimDurList,
                     'dataType': dataTypeList})

sns.barplot(data=ratioData, x="stimDur", y="data", hue="dataType")