"""Compare times to peak across layers"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
    for modality in ['vaso', 'bold']:
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
                        if maxVal != 0 and val <= 0:
                            break
                        if val >= maxVal:
                            maxTP = timePoint
                            maxVal = val

                    layerList.append(layerNames[int(layer-1)])
                    valList.append(maxVal)
                    stimDurList.append(stimDur)
                    dataTypeList.append(dataType)
                    timePointList.append(maxTP)
                    subList.append(sub)
                    modalityList.append(modality)

peakTimeList = [i * 0.785 for i in timePointList]

timepointData = pd.DataFrame({'subject': subList,
                              'maxVol': timePointList,
                              'data': valList,
                              'layer': layerList,
                              'stimDur': stimDurList,
                              'dataType': dataTypeList,
                              'peakTime': peakTimeList,
                              'modality': modalityList})

timepointData.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/timeToPeak.csv',
                     sep=',',
                     index=False)

timepointData = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/timeToPeak.csv',
                            sep=',')

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


# Barplots
yMax = {'bold': [6, 7, 8, 15, 19], 'vaso': [7, 7, 10, 15, 24]}

for modality in ['vaso', 'bold']:
    fig, axes = plt.subplots(1, 5, figsize=(21, 5), sharey=True)

    for i, stimDur in enumerate(timepointData['stimDur'].unique()):

        tmp = timepointData.loc[(timepointData['dataType'] == 'raw')
                                & (timepointData['stimDur'] == stimDur)
                                & (timepointData['modality'] == modality)
                                & (timepointData['subject'] != 'sub-08')]

        sns.barplot(ax=axes[i], data=tmp, x="stimDur", y="peakTime", hue="layer", palette=palettesLayers[modality])

        # ================================================================================
        # Misc
        # axes[i].set_ylim(0, yMax[modality][i])
        #
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

        # Legend
        if i < 4:
            axes[i].get_legend().remove()
        if i > 0:
            axes[i].set(ylabel=None)

        if i == 0:
            axes[i].set_ylabel('Time to peak [s]', fontsize=18)

        axes[i].axhline(stimDur, linestyle='--', color='white')

        legend = plt.legend(title='Layer', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
        title = legend.get_title()
        title.set_fontsize(18)
    plt.savefig(
        f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/TTP_{modality}.png',
        bbox_inches="tight")
    plt.show()


# Line graphs
for modality in ['vaso', 'bold']:
    fig, axes = plt.subplots(1, 5, figsize=(21, 5))

    for i, stimDur in enumerate(timepointData['stimDur'].unique()):
        tmp = timepointData.loc[(timepointData['dataType'] == 'raw')
                                & (timepointData['stimDur'] == stimDur)
                                & (timepointData['modality'] == modality)]
        print(tmp)
        sns.lineplot(ax=axes[i], data=tmp, x="layer", y="peakTime", hue="subject")

    plt.show()

# ============================================================================================================
# Stats

# Read an example dataset
import pingouin as pg

for modality in ['vaso', 'bold']:
    for i, stimDur in enumerate(timepointData['stimDur'].unique()):
        print('')
        print(f'{modality} stimulus duration: {stimDur}')

        tmp = timepointData.loc[(timepointData['dataType'] == 'raw')
                                & (timepointData['stimDur'] == stimDur)
                                & (timepointData['modality'] == modality)
                                & (timepointData['subject'] != 'sub-08')]

        aov = pg.rm_anova(data=tmp, dv='peakTime', subject='subject', within='layer', detailed=False, effsize='np2')

        if aov['p-unc'][0] <= 0.05:
            print(aov)
            post_hocs = pg.pairwise_tests(dv='peakTime', within='layer', subject='subject', effsize='eta-square', data=tmp)
            print(post_hocs)
            sig = post_hocs

            sig['T']
            sig['dof']
