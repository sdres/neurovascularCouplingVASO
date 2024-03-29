"""

Extract statistical values from layer ROIs

"""

import pandas as pd
import numpy as np
import nibabel as nb
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Set data path
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'
OUTDIR = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/results'

# Set subjects to work on
subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
# subs = ['sub-06','sub-07']
# subs = ['sub-08']

MODALITIES = ['bold', 'vaso']

subList = []
depthList = []
valList = []
stimList = []
modalityList = []

for sub in subs:
    print(f'Processing {sub}')

    statFolder = f'{DATADIR}/{sub}/statMaps/glm_fsl'
    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    depthFile = glob.glob(f'{segFolder}/{sub}_rim-LH*layers_equivol.nii*')[0]
    depthNii = nb.load(depthFile)
    depthData = depthNii.get_fdata()
    layers = np.unique(depthData)[1:]

    roisData = nb.load(glob.glob(f'{segFolder}/{sub}_rim-LH_perimeter_chunk.nii.gz')[0]).get_fdata()
    roiIdx = roisData == 1

    for stimDuration in [1, 2, 4, 12, 24]:
        for modality in MODALITIES:

            stimData = nb.load(glob.glob(f'{statFolder}/{sub}_{modality}_stim_{stimDuration}s_registered_crop-toShpereLH.nii.gz')[0]).get_fdata()

            for layer in layers:

                layerIdx = depthData == layer
                tmp = roiIdx*layerIdx

                val = np.mean(stimData[tmp])

                subList.append(sub)
                depthList.append(layer)
                valList.append(val)
                stimList.append(stimDuration)
                modalityList.append(modality)

# Save data to dataframe
data = pd.DataFrame({'subject': subList,
                     'depth': depthList,
                     'value': valList,
                     'stim': stimList,
                     'modality': modalityList}
                    )

data.to_csv(f'{OUTDIR}/zscoreData.csv',
            sep=',',
            index=False)

# ===========================================================================
# Plotting
# ===========================================================================

data = pd.read_csv(f'{OUTDIR}/zscoreData.csv', sep=',')

data = data.loc[data['subject'] != 'sub-08']

plt.style.use('dark_background')

palettes = {
    'bold': ['#ff7f0e', '#ff9436', '#ffaa5e', '#ffbf86', '#ffd4af'],
    'vaso': ['#1f77b4', '#2a92da', '#55a8e2', '#7fbee9', '#aad4f0']}

for modality in ['bold', 'vaso']:
    fig, ax = plt.subplots()

    tmp = data.loc[(data['modality'] == modality)]

    sns.lineplot(data=tmp, x='depth', y='value', hue='stim', linewidth=2, palette=palettes[modality])

    # Set y ticks and axis
    lim = ax.get_ylim()
    ax.set_yticks(np.linspace(lim[0].round(), lim[1].round(), 5).round(1))
    plt.yticks(fontsize=18)
    plt.ylabel(f'Z-score', fontsize=20)

    # plt.title(f"{modality}", fontsize=24, pad=20)

    # Set x ticks and axis
    plt.xticks([1, 11], fontsize=18)
    ax.set_xticklabels(['WM', 'CSF'])
    plt.xlabel('')

    plt.legend(loc='upper left')
    legend = plt.legend(title='Stim dur [s]', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

    title = legend.get_title()
    title.set_fontsize(14)
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/sub-all_{modality}_zScoreProfile.png', bbox_inches="tight")
    plt.show()


for sub in subs:
    for modality in ['bold', 'vaso']:
        tmp = data.loc[(data['modality'] == modality) & (data['subject'] == sub)]

        fig, ax = plt.subplots()  # Initiate plot
        sns.lineplot(data=tmp, x='depth', y='value', hue='stim', linewidth=2, palette=palettes[modality])  # Plot data

        # Set x ticks and axis
        plt.xticks([1, 11], fontsize=18)
        ax.set_xticklabels(['WM', 'CSF'])
        plt.xlabel('')

        # Set y ticks and axis
        if modality == 'vaso':
            upper = 7
            lower = 0
            ax.set_yticks(np.arange(lower, upper, 1))
        if modality == 'bold':
            upper = 28
            lower = 0
            ax.set_yticks(np.arange(lower, upper, 3))

        plt.ylim(lower, upper-1)

        plt.ylabel(f'z-score', fontsize=20)
        plt.yticks(fontsize=18)

        # Set legend
        plt.legend(loc='upper left')
        legend = plt.legend(title='Stim dur [s]',
                            fontsize=14,
                            loc='center left',
                            bbox_to_anchor=(1, 0.5)
                            )

        # Set title
        plt.title(f"{sub}", fontsize=24, pad=20)
        title = legend.get_title()
        title.set_fontsize(14)

        # Save figure
        plt.tight_layout()
        plt.savefig(f'{OUTDIR}/layerProfiles/{sub}_{modality}_zScoreProfile.png', bbox_inches="tight")
        plt.show()

# ===========================================================================
# Normalize mean profiles to show peak locations
# ===========================================================================

# Get mean profile across subjects for each stimulus duration
layerList = []
stimDurList = []
normValList = []
modalityList = []

for stimDuration in [1, 2, 4, 12, 24]:

    for modality in MODALITIES:
        valList = []

        for layer in data['depth'].unique():

            tmp = data.loc[(data['modality'] == modality) & (data['stim'] == stimDuration) & (data['depth'] == layer)]

            mean = np.mean(tmp['value'])
            valList.append(mean)

        minVal = np.min(valList)
        maxVal = np.max(valList)

        normVals = [((val-minVal) / (maxVal-minVal)) for val in valList]

        for i, val in enumerate(normVals):
            normValList.append(val)
            layerList.append(data['depth'].unique()[i])
            stimDurList.append(stimDuration)
            modalityList.append(modality)


# Save data to dataframe
dataNorm = pd.DataFrame({
                     'depth': layerList,
                     'normVals': normValList,
                     'stim': stimDurList,
                     'modality': modalityList}
                    )

# Plot group
for modality in ['vaso', 'bold']:

    tmp = dataNorm.loc[(dataNorm['modality'] == modality)]

    fig, ax = plt.subplots()

    sns.lineplot(data=tmp,
                 x='depth',
                 y='normVals',
                 hue='stim',
                 linewidth=2,
                 palette=palettes[modality]
                 )

    plt.ylabel(f'Z-score [norm.]', fontsize=20)
    plt.yticks(fontsize=18)

    # Set x ticks and axis
    plt.xticks([1, 11], fontsize=18)
    ax.set_xticklabels(['WM', 'CSF'])
    plt.xlabel('')

    plt.legend(loc='upper left')
    legend = plt.legend(title='Stim dur [s]', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

    title = legend.get_title()
    title.set_fontsize(14)
    plt.tight_layout()
    plt.savefig(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/group_{modality}_normalizedProfiles.png', bbox_inches="tight")
    plt.show()


# ===========================================================================
# Normalize profiles to show peak locations
# ===========================================================================

palettes = {
    'bold': ['#ff7f0e', '#ff9436', '#ffaa5e', '#ffbf86', '#ffd4af'],
    'vaso': ['#1f77b4', '#2a92da', '#55a8e2', '#7fbee9', '#aad4f0']}

normData = pd.DataFrame()

for modality in ['vaso']:
    for stimDur in data['stim'].unique():
        for sub in subs:

            tmp = data.loc[(data['modality'] == modality) & (data['stim'] == stimDur) & (data['subject'] == sub)]

            minVal = np.min(tmp['value'])
            maxVal = np.max(tmp['value'])

            normVals = [((val-minVal) / (maxVal-minVal)) for val in tmp['value']]
            tmp['normVals'] = normVals

            normData = pd.concat([normData, tmp])


# Plot individual subjects
for modality in ['vaso']:
    for sub in subs:

        fig, ax = plt.subplots()

        sns.lineplot(data=normData.loc[normData['subject'] == sub],
                     x='depth',
                     y='normVals',
                     hue='stim',
                     linewidth=2,
                     palette=palettes[modality]
                     )

        plt.ylabel(f'Z-score [norm.]', fontsize=20)
        plt.yticks(fontsize=18)

        # Set x ticks and axis
        plt.xticks([1, 11], fontsize=18)
        ax.set_xticklabels(['WM', 'CSF'])
        plt.xlabel('')

        plt.legend(loc='upper left')
        legend = plt.legend(title='Stim dur [s]', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.title(f"{sub}", fontsize=24, pad=20)
        title = legend.get_title()
        title.set_fontsize(14)
        plt.tight_layout()
        plt.savefig(f'../results/{sub}_{modality}_normalizedProfiles.png', bbox_inches="tight")
        plt.show()


# Plot group
for modality in ['vaso']:

    fig, ax = plt.subplots()

    sns.lineplot(data=normData.loc[normData['subject'] != 'sub-08'],
                 x='depth',
                 y='normVals',
                 hue='stim',
                 linewidth=2,
                 ci=None,
                 palette=palettes[modality]
                 )

    plt.ylabel(f'Z-score [norm.]', fontsize=20)
    plt.yticks(fontsize=18)

    # Set x ticks and axis
    plt.xticks([1, 11], fontsize=18)
    ax.set_xticklabels(['WM', 'CSF'])
    plt.xlabel('')

    plt.legend(loc='upper left')
    legend = plt.legend(title='Stim dur [s]', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(f"Normalized group VASO", fontsize=24, pad=20)
    title = legend.get_title()
    title.set_fontsize(14)
    plt.tight_layout()
    plt.savefig(f'../results/group_{modality}_normalizedProfiles.png', bbox_inches="tight")
    plt.show()

# ===========================================================================
# Normalize mean profiles to show peak locations for individual subjects
# ===========================================================================

# Get mean profile across subjects for each stimulus duration
layerList = []
stimDurList = []
normValList = []
modalityList = []
subList = []

for sub in data['subject'].unique():
    for stimDuration in [1, 2, 4, 12, 24]:

        for modality in MODALITIES:
            valList = []

            for layer in data['depth'].unique():

                tmp = data.loc[(data['modality'] == modality) & (data['stim'] == stimDuration) & (data['depth'] == layer) & (data['subject'] == sub)]
                # tmp = data.loc[(data['modality'] == modality) & (data['stim'] == stimDuration) & (data['depth'] == layer)]

                mean = np.mean(tmp['value'])
                valList.append(mean)

            minVal = np.min(valList)
            maxVal = np.max(valList)

            normVals = [((val-minVal) / (maxVal-minVal)) for val in valList]

            for i, val in enumerate(normVals):
                normValList.append(val)
                layerList.append(data['depth'].unique()[i])
                stimDurList.append(stimDuration)
                modalityList.append(modality)
                subList.append(sub)

# Save data to dataframe
dataNorm = pd.DataFrame({
                     'depth': layerList,
                     'normVals': normValList,
                     'stim': stimDurList,
                     'modality': modalityList,
                     'subject': subList})

palettes = {
    'bold': ['#ff7f0e', '#ff9436', '#ffaa5e', '#ffbf86', '#ffd4af'],
    'vaso': ['#1f77b4', '#2a92da', '#55a8e2', '#7fbee9', '#aad4f0']}

# Plot individual subjects
for modality in ['vaso']:
    for sub in subs:

        fig, ax = plt.subplots()

        sns.lineplot(data=dataNorm.loc[(dataNorm['subject'] == sub) & (dataNorm['modality'] == modality)],
                     x='depth',
                     y='normVals',
                     hue='stim',
                     linewidth=2,
                     palette=palettes[modality]
                     )

        plt.ylabel(f'Z-score [norm.]', fontsize=20)
        plt.yticks(fontsize=18)

        # Set x ticks and axis
        plt.xticks([1, 11], fontsize=18)
        ax.set_xticklabels(['WM', 'CSF'])
        plt.xlabel('')

        plt.legend(loc='upper left')
        legend = plt.legend(title='Stim dur [s]', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.title(f"{sub}", fontsize=24, pad=20)
        title = legend.get_title()
        title.set_fontsize(14)
        plt.tight_layout()
        plt.savefig(f'/Users/sebastiandresbach/Desktop/{sub}_{modality}_normalizedProfiles.png', bbox_inches="tight")
        plt.show()


# Plot individual subjects
for modality in ['vaso']:

    fig, ax = plt.subplots()

    sns.lineplot(data=dataNorm.loc[(dataNorm['modality'] == modality)],
                 x='depth',
                 y='normVals',
                 hue='stim',
                 linewidth=2,
                 ci=None,
                 palette=palettes[modality]
                 )

    plt.ylabel(f'Z-score [norm.]', fontsize=20)
    plt.yticks(fontsize=18)

    # Set x ticks and axis
    plt.xticks([1, 11], fontsize=18)
    ax.set_xticklabels(['WM', 'CSF'])
    plt.xlabel('')

    plt.legend(loc='upper left')
    legend = plt.legend(title='Stim dur [s]', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(f"Group (n=5)", fontsize=24, pad=20)
    title = legend.get_title()
    title.set_fontsize(14)
    plt.tight_layout()
    plt.savefig(f'/Users/sebastiandresbach/Desktop/Group_{modality}_normalizedProfiles_noErr.png', bbox_inches="tight")
    plt.show()

# ==========================================================================================
# Only take positive Voxs
# ==========================================================================================

subList = []
depthList = []
valList = []
stimList = []
modalityList = []

subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']

# subs = ['sub-06']

for sub in subs:
    print(f'Processing {sub}')

    statFolder = f'{DATADIR}/{sub}/statMaps/glm_fsl'
    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    depthFile = glob.glob(f'{segFolder}/{sub}_rim-LH*layers_equivol.nii*')[0]
    depthNii = nb.load(depthFile)
    depthData = depthNii.get_fdata()
    layers = np.unique(depthData)[1:]

    roisData = nb.load(glob.glob(f'{segFolder}/{sub}_rim-LH_perimeter_chunk.nii*')[0]).get_fdata()
    roiIdx = roisData == 1

    for stimDuration in [1, 2, 4, 12, 24]:
        for modality in MODALITIES:

            stimData = nb.load(glob.glob(f'{statFolder}/{sub}_{modality}_stim_{stimDuration}s_registered_crop-toShpereLH.nii.gz')[0]).get_fdata()

            for layer in layers:

                layerIdx = depthData == layer
                tmp = roiIdx*layerIdx

                tmpPos = stimData >= 2.5
                tmp = tmp * tmpPos

                val = np.mean(stimData[tmp])

                subList.append(sub)
                depthList.append(layer)
                valList.append(val)
                stimList.append(stimDuration)
                modalityList.append(modality)

# Save data to dataframe
data = pd.DataFrame({'subject': subList,
                     'depth': depthList,
                     'value': valList,
                     'stim': stimList,
                     'modality': modalityList}
                    )

palettes = {
    'bold': ['#ff7f0e', '#ff9436', '#ffaa5e', '#ffbf86', '#ffd4af'],
    'vaso': ['#1f77b4', '#2a92da', '#55a8e2', '#7fbee9', '#aad4f0']}

for sub in subs:
    for modality in ['bold', 'vaso']:
        fig, ax = plt.subplots()

        tmp = data.loc[(data['modality'] == modality) & (data['subject'] == sub)]

        sns.lineplot(data=tmp, x='depth', y='value', hue='stim', linewidth=2, palette=palettes[modality])

        # Set y ticks and axis
        lim = ax.get_ylim()
        ax.set_yticks(np.linspace(lim[0].round(), lim[1].round(), 5).round(1))
        plt.yticks(fontsize=18)
        plt.ylabel(f'Z-score', fontsize=20)

        # plt.title(f"{modality}", fontsize=24, pad=20)

        # Set x ticks and axis
        plt.xticks([1, 11], fontsize=18)
        ax.set_xticklabels(['WM', 'CSF'])
        plt.xlabel('')

        plt.legend(loc='upper left')
        legend = plt.legend(title='Stim dur [s]', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

        title = legend.get_title()
        title.set_fontsize(14)
        plt.tight_layout()
        plt.savefig(f'{OUTDIR}/{sub}_{modality}_zScoreProfile_positiveOnly.png', bbox_inches="tight")
        plt.show()


# =================================
# Investigate relation between activation strength across stimulus durations

data = pd.read_csv(f'{OUTDIR}/zscoreData.csv', sep=',')

data = data.loc[data['subject'] != 'sub-08']

plt.style.use('dark_background')
palette = {
    'bold': 'tab:orange',
    'vaso': 'tab:blue'}
subList = []
stimList = []
modalityList = []
maxValList = []

for sub in data['subject'].unique():

    for stimDuration in data['stim'].unique():

        for modality in data['modality'].unique():
            tmp = data.loc[(data['subject'] == sub)
                           & (data['stim'] == stimDuration)
                           & (data['modality'] == modality)]

            # Find max vale
            maxVal = np.amax(tmp['value'])

            subList.append(sub)
            stimList.append(stimDuration)
            modalityList.append(modality)
            maxValList.append(maxVal)

data = pd.DataFrame({'subject': subList,
                     'value': maxValList,
                     'stim': stimList,
                     'modality': modalityList}
                    )

vals = []

fig, ax = plt.subplots()
splot = sns.barplot(ax= ax, data=data, x='stim', y='value', hue='modality', palette=palette)

for stimDuration in data['stim'].unique():
    for modality in data['modality'].unique():
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.2f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points',
                           fontsize=14)
            if modality == 'vaso':
                vals.append(p.get_height())
plt.show()

# =================================
# Investigate sub-08
stimList = []
modalityList = []
slopeList = []

palette = {
    'bold': 'tab:orange',
    'vaso': 'tab:blue'}

for stimDuration in data['stim'].unique():
    fig, ax = plt.subplots()

    for modality in data['modality'].unique():
        tmp = data.loc[(data['subject'] == 'sub-08')
                       & (data['stim'] == stimDuration)
                       & (data['modality'] == modality)]

        # find line of best fit
        a, b = np.polyfit(tmp['depth'], tmp['value'], 1)

        # add points to plot
        plt.scatter(tmp['depth'], tmp['value'], label=f'{modality} data', color=palette[modality])
        plt.title(f'stimdur: {stimDuration}')

        # add line of best fit to plot
        plt.plot(tmp['depth'], a * tmp['depth'] + b, label=f'{modality} fit', linestyle='--', linewidth=2, color=palette[modality])

        stimList.append(stimDuration)
        modalityList.append(modality)
        slopeList.append(a)

    plt.yticks(fontsize=18)
    plt.ylabel(f'Z-score', fontsize=20)
    # Set x ticks and axis
    plt.xticks([1, 11], fontsize=18)
    ax.set_xticklabels(['WM', 'CSF'])
    plt.xlabel('')

    plt.legend()
    plt.savefig(
        f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/fit_{stimDuration}.png',
        bbox_inches="tight")
    plt.show()

slopeData = pd.DataFrame({'stimDur': stimList, 'modality': modalityList, "slope": slopeList})


fig, ax = plt.subplots(1, 1)
sns.barplot(data=slopeData, x="stimDur", y="slope", hue="modality", palette=palette)
plt.savefig(
    f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/slope.png',
    bbox_inches="tight")
plt.show()

stimList = []
modalityList = []
ratioList = []

for stimDuration in data['stim'].unique():

    for modality in data['modality'].unique():
        tmp = data.loc[(data['subject'] == 'sub-08')
                       & (data['stim'] == stimDuration)
                       & (data['modality'] == modality)]

        val1 = np.mean(tmp['value'].to_numpy()[-2:])
        val2 = np.mean(tmp['value'].to_numpy()[:2])
        print(f'{modality} {stimDuration}')
        print(f'{val1} {val2}')

        val = np.mean(tmp['value'].to_numpy()[-2:]) / np.mean(tmp['value'].to_numpy()[:2])

        stimList.append(stimDuration)
        modalityList.append(modality)
        ratioList.append(val)

ratioData = pd.DataFrame({'stimDur': stimList, 'modality': modalityList, "ratio": ratioList})

fig, ax = plt.subplots(1, 1)
sns.barplot(data=ratioData, x="stimDur", y="ratio", hue="modality", palette=palette)
plt.savefig(
    f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ratio.png',
    bbox_inches="tight")
plt.show()
