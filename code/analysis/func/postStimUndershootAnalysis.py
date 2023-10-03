"""Investigate the post stimulus undershoot across cortical depth"""


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalizedNormalized.csv', sep=',')

layerNames = ['deep', 'middle', 'superficial', 'vein']

layerList = []
peakValList = []
undershootValList = []
normUndershootValList = []
stimDurList = []
peakTimePointList = []
undershootTimePointList = []
subList = []
modalityList = []

for sub in data['subject'].unique():
    for modality in ['vaso', 'bold']:
        for stimDur in data['stimDur'].unique():
            for layer in data['layer'].unique():

                tmp = data.loc[(data['dataType'] == 'raw')
                               & (data['subject'] == sub)
                               & (data['modality'] == modality)
                               & (data['layer'] == layer)
                               & (data['stimDur'] == stimDur)]

                maxTP = 0
                maxVal = 0

                # Find timepoint with highest value
                for timePoint in tmp['volume'].unique():
                    val = np.mean(tmp.loc[tmp['volume'] == timePoint]['data'])
                    if maxVal != 0 and val <= 0:
                        break
                    if val >= maxVal:
                        maxTP = timePoint
                        maxVal = val

                stimCess = stimDur/0.785
                lowest = 10
                underTP = 0
                # Find lowest point after stimulus cessation
                for timePoint in tmp['volume'].unique():
                    if timePoint >= stimCess:
                        val = np.mean(tmp.loc[tmp['volume'] == timePoint]['data'])
                        if val <= lowest:
                            lowest = val
                            underTP = timePoint

                # Normalize undershoot by peak height
                normedUndershoot = lowest / maxVal

                layerList.append(layerNames[int(layer-1)])
                peakValList.append(maxVal)
                undershootValList.append(lowest)
                normUndershootValList.append(normedUndershoot)
                stimDurList.append(stimDur)
                peakTimePointList.append(maxTP)
                undershootTimePointList.append(underTP)
                subList.append(sub)
                modalityList.append(modality)

peakTimeList = [i * 0.785 for i in peakTimePointList]
undershootTimeList = [i * 0.785 for i in undershootTimePointList]

temporalData = pd.DataFrame({'subject': subList,
                              'peakVol': peakTimeList,
                              'peak': peakValList,
                              'layer': layerList,
                              'stimDur': stimDurList,
                              'peakTime': peakTimeList,
                             'undershootVal': undershootValList,
                             'undershootValNorm': normUndershootValList,
                             'undershootTime': undershootTimePointList,
                              'modality': modalityList})


# ============================================================================================================
# Plotting
# ============================================================================================================

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

lims = []


for dataType in ['undershootVal', 'undershootValNorm']:
    for modality in ['vaso', 'bold']:

        # Get limits of 24s stimDur
        tmp = temporalData.loc[(temporalData['modality'] == modality)
                               & (temporalData['subject'] != 'sub-08')]

        ymin = np.min(tmp[dataType])


        fig, axes = plt.subplots(1, 5, figsize=(21, 5), sharey=True)

        for i, stimDur in enumerate(temporalData['stimDur'].unique()):

            tmp = temporalData.loc[(temporalData['stimDur'] == stimDur)
                                    & (temporalData['modality'] == modality)
                                    & (temporalData['subject'] != 'sub-08')]

            sns.barplot(ax=axes[i], data=tmp, x="stimDur", y=dataType, hue="layer", palette=palettesLayers[modality])

            # sns.barplot(ax=axes[i], data=tmp, x="stimDur", y="undershootVal", hue="layer", palette=palettesLayers[modality], errorbar=None)

            # ================================================================================
            # Mis
            # if modality == 'bold':
            if modality == 'vaso' and dataType == 'undershootValNorm':
                axes[i].set_ylim(-1, 0)
            else:
                axes[i].set_ylim(ymin, 0)

            if stimDur == 1:
                axes[i].set_title(f'{int(stimDur)} second stimulation', fontsize=18, pad=titlePad)
            else:
                axes[i].set_title(f'{int(stimDur)} seconds stimulation', fontsize=18, pad=titlePad)

            # Set font-sizes for axes
            axes[i].yaxis.set_tick_params(labelsize=18)
            axes[i].set(xlabel=None)
            axes[i].set_xticks([])

            # Legend
            if i < 4:
                axes[i].get_legend().remove()
            if i > 0:
                axes[i].set(ylabel=None)

            if i == 0 and dataType == 'undershootVal':
                axes[i].set_ylabel('PSU [% signal change]', fontsize=18)
            elif i == 0 and dataType == 'undershootValNorm':
                axes[i].set_ylabel('Normalized PSU [a.u.]', fontsize=18)
            # handle the duplicate legend
            handles, labels = axes[i].get_legend_handles_labels()

            # legend = plt.legend(handles[-2:], labels[-2:], title='Layer', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
            legend = plt.legend(title='Layer', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
            title = legend.get_title()
            title.set_fontsize(18)
        plt.savefig(
            f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/PSU_{modality}_{dataType}.png',
            bbox_inches="tight")
        plt.show()
