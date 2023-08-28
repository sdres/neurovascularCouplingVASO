"""Analysis of onset times across layers"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sympy import *


# Load data
data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalizedNormalized.csv', sep=',')

# Average data across participants
layerList = []
valList = []
stimDurList = []
timePointList = []
modalityList = []

for modality in ['vaso', 'bold']:
    for stimDur in data['stimDur'].unique():
        for layer in data['layer'].unique():
            for volume in data['volume'].unique():
                tmp = data.loc[(data['dataType'] == 'raw')
                               & (data['subject'] != 'sub-08')
                               & (data['modality'] == modality)
                               & (data['layer'] == layer)
                               & (data['stimDur'] == stimDur)
                               & (data['volume'] == volume)]

                val = np.mean(tmp['data'])

                layerList.append(layer)
                valList.append(val)
                stimDurList.append(stimDur)
                timePointList.append(volume)
                modalityList.append(modality)

avgData = pd.DataFrame({'data': valList,
                        'layer': layerList,
                        'stimDur': stimDurList,
                        'modality': modalityList,
                        'volume': timePointList})

layerNames = ['deep', 'middle', 'superficial', 'vein']

layerList = []
valList = []
stimDurList = []
dataTypeList = []
timePointList = []
modalityList = []


for modality in ['vaso', 'bold']:
    for stimDur in data['stimDur'].unique():
        for layer in data['layer'].unique():

            tmp = data.loc[(data['dataType'] == 'raw')
                           & (data['subject'] != 'sub-08')
                           & (data['modality'] == modality)
                           & (data['layer'] == layer)
                           & (data['stimDur'] == stimDur)]

            firstPeakTP = 0
            firstPeakVal = 0

            # Find first peak
            for timePoint in tmp['volume'].unique():
                val = np.mean(tmp.loc[tmp['volume'] == timePoint]['data'])

                if firstPeakVal != 0 and val < firstPeakVal:
                    break

                if val >= firstPeakVal:
                    firstPeakTP = timePoint
                    firstPeakVal = val

            # Alright, now we have the first peak, and it's value. So now we can compute the timepoints where the
            # values are in the range of 20-80% of this.

            lowBound = (firstPeakVal/100) * 20
            upBound = (firstPeakVal / 100) * 80

            tmp2 = avgData.loc[(avgData['modality'] == modality)
                           & (avgData['layer'] == layer)
                           & (avgData['stimDur'] == stimDur)
                           & (avgData['data'] >= lowBound)
                           & (avgData['data'] <= upBound)
                           & (avgData['volume'] <= firstPeakTP)]

            # Fit slope
            a, b = np.polyfit(tmp2['volume'], tmp2['data'], 1)

            x, y = symbols('x y')

            y = x * a + b

            # set the expression, y, equal to 0 and solve
            result = solve(Eq(y, 0))

            layerList.append(layerNames[int(layer-1)])
            valList.append(result[0])
            stimDurList.append(stimDur)
            modalityList.append(modality)


onsetData = pd.DataFrame({'data': valList,
                          'layer': layerList,
                          'stimDur': stimDurList,
                          'modality': modalityList})

onsetData.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/onsetTimeAnalysis.csv',
                     sep=',',
                     index=False)

palettesLayers = {'vaso': ['#55a8e2', '#aad4f0', '#ffffff', '#FF0000'],
                  'bold': ['#ff8c26', '#ffd4af', '#ffffff', '#FF0000']}

plt.style.use('dark_background')

# define linewidth to 2
LW = 2
# Define fontsize size for x- and y-labels
labelSize = 24
# Define fontsize size for x- and y-ticks
tickLabelSize = 18
# Define fontsize legend text
legendTextSize = 18
titlePad = 10


# Barplots
yMax = {'bold': [6, 7, 8, 15, 19], 'vaso': [7, 7, 10, 15, 24]}

for modality in ['vaso', 'bold']:
    fig, axes = plt.subplots(1, 5, figsize=(21, 5))

    for i, stimDur in enumerate(onsetData['stimDur'].unique()):

        tmp = onsetData.loc[(onsetData['stimDur'] == stimDur)
                            & (onsetData['modality'] == modality)]

        sns.barplot(ax=axes[i], data=tmp, x="stimDur", y="data", hue="layer", palette=palettesLayers[modality])

        # ================================================================================
        # Misc
        # axes[i].set_ylim(0, yMax[modality][i])

        # ticks = np.arange(0, yMax[modality][i], int(yMax[modality][i]/6))
        # axes[i].set_yticks(ticks)

        if stimDur == 1:
            axes[i].set_title(f'{int(stimDur)} second stimulation', fontsize=18, pad=titlePad)
        else:
            axes[i].set_title(f'{int(stimDur)} seconds stimulation', fontsize=18, pad=titlePad)

        # Set font-sizes for axes
        axes[i].yaxis.set_tick_params(labelsize=18)
        axes[i].set(xlabel=None)
        axes[i].set_xticks([])

        # axes[i].axhline(stimDur, linestyle='--', color='white')

        # Legend
        if i < 4:
            axes[i].get_legend().remove()
        if i > 0:
            axes[i].set(ylabel=None)

        if i == 0:
            axes[i].set_ylabel('Time to peak [s]', fontsize=18)

        legend = plt.legend(title='Layer', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
        title = legend.get_title()
        title.set_fontsize(18)
    # plt.savefig(
    #     f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/TTP_{modality}.png',
    #     bbox_inches="tight")
    plt.show()

tmp = timepointData.loc[(timepointData['dataType'] == 'raw')
                        & (timepointData['stimDur'] == 4)
                        & (timepointData['modality'] == 'vaso')
                        & (timepointData['subject'] == 'sub-07')]




# saved for later






# Load data
data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalizedNormalized.csv', sep=',')

layerNames = ['deep', 'middle', 'superficial', 'vein']

layerList = []
valList = []
stimDurList = []
dataTypeList = []
timePointList = []
subList = []
modalityList = []

for sub in data['subject'].unique():
    if sub == 'sub-08':
        continue
    if sub == 'sub-05':
        continue
    for modality in ['vaso', 'bold']:
        for stimDur in data['stimDur'].unique():
            for layer in data['layer'].unique():

                tmp = data.loc[(data['dataType'] == 'raw')
                               & (data['subject'] == sub)
                               & (data['modality'] == modality)
                               & (data['layer'] == layer)
                               & (data['stimDur'] == stimDur)]

                firstPeakTP = 0
                firstPeakVal = 0

                # Find first peak
                for timePoint in tmp['volume'].unique():
                    val = np.mean(tmp.loc[tmp['volume'] == timePoint]['data'])

                    if firstPeakVal != 0 and val < firstPeakVal:
                        break

                    if val >= firstPeakVal:
                        firstPeakTP = timePoint
                        firstPeakVal = val

                # Alright, now we have the first peak, and it's value. So now we can compute the timepoints where the
                # values are in the range of 20-80% of this.

                lowBound = (firstPeakVal/100) * 20
                upBound = (firstPeakVal / 100) * 80

                tmp2 = data.loc[(data['dataType'] == 'raw')
                               & (data['subject'] == sub)
                               & (data['modality'] == modality)
                               & (data['layer'] == layer)
                               & (data['stimDur'] == stimDur)
                               & (data['data'] >= lowBound)
                               & (data['data'] <= upBound)
                               & (data['volume'] <= firstPeakTP)]

                # Fit slope
                a, b = np.polyfit(tmp2['volume'], tmp2['data'], 1)

                x, y = symbols('x y')

                y = x * a + b

                # set the expression, y, equal to 0 and solve
                result = solve(Eq(y, 0))

                layerList.append(layerNames[int(layer-1)])
                valList.append(result[0])
                stimDurList.append(stimDur)
                dataTypeList.append(dataType)
                subList.append(sub)
                modalityList.append(modality)


timepointData = pd.DataFrame({'subject': subList,
                              'data': valList,
                              'layer': layerList,
                              'stimDur': stimDurList,
                              'dataType': dataTypeList,
                              'modality': modalityList})

timepointData.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/onsetTimeAnalysis.csv',
                     sep=',',
                     index=False)

# Alright, now we have the first peak, and it's value. So now we can compute the timepoints where the values are in the
# range of 20-80% of this.








palettesLayers = {'vaso': ['#55a8e2', '#aad4f0', '#ffffff', '#FF0000'],
                  'bold': ['#ff8c26', '#ffd4af', '#ffffff', '#FF0000']}

plt.style.use('dark_background')

# define linewidth to 2
LW = 2
# Define fontsize size for x- and y-labels
labelSize = 24
# Define fontsize size for x- and y-ticks
tickLabelSize = 18
# Define fontsize legend text
legendTextSize = 18
titlePad = 10


# Barplots
yMax = {'bold': [6, 7, 8, 15, 19], 'vaso': [7, 7, 10, 15, 24]}

for modality in ['vaso', 'bold']:
    fig, axes = plt.subplots(1, 5, figsize=(21, 5))

    for i, stimDur in enumerate(timepointData['stimDur'].unique()):

        tmp = timepointData.loc[(timepointData['dataType'] == 'raw')
                                & (timepointData['stimDur'] == stimDur)
                                & (timepointData['modality'] == modality)]

        sns.barplot(ax=axes[i], data=tmp, x="stimDur", y="peakTime", hue="layer", palette=palettesLayers[modality])

        # ================================================================================
        # Misc
        axes[i].set_ylim(0, yMax[modality][i])

        ticks = np.arange(0, yMax[modality][i], int(yMax[modality][i]/6))
        axes[i].set_yticks(ticks)

        if stimDur == 1:
            axes[i].set_title(f'{int(stimDur)} second stimulation', fontsize=18, pad=titlePad)
        else:
            axes[i].set_title(f'{int(stimDur)} seconds stimulation', fontsize=18, pad=titlePad)

        # Set font-sizes for axes
        axes[i].yaxis.set_tick_params(labelsize=18)
        axes[i].set(xlabel=None)
        axes[i].set_xticks([])

        # axes[i].axhline(stimDur, linestyle='--', color='white')

        # Legend
        if i < 4:
            axes[i].get_legend().remove()
        if i > 0:
            axes[i].set(ylabel=None)

        if i == 0:
            axes[i].set_ylabel('Time to peak [s]', fontsize=18)

        legend = plt.legend(title='Layer', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
        title = legend.get_title()
        title.set_fontsize(18)
    # plt.savefig(
    #     f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/TTP_{modality}.png',
    #     bbox_inches="tight")
    plt.show()

tmp = timepointData.loc[(timepointData['dataType'] == 'raw')
                        & (timepointData['stimDur'] == 4)
                        & (timepointData['modality'] == 'vaso')
                        & (timepointData['subject'] == 'sub-07')]

