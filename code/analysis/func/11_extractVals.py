'''

Extract statistical values from layer ROIs

'''

import subprocess
import pandas as pd
import numpy as np
import nibabel as nb
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Set data path
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# Set subjects to work on
subs = ['sub-05','sub-06','sub-07','sub-09']
# subs = ['sub-06','sub-07']
subs = ['sub-09']

MODALITIES = ['bold', 'vaso']


subList = []
depthList = []
valList = []
stimList = []
modalityList = []

for sub in subs:
    print(f'Processing {sub}')

    statFolder = f'{DATADIR}/{sub}/statMaps'
    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    # roiFolder = f'{DATADIR}/{sub}/rois'


    depthFile = glob.glob(f'{segFolder}/{sub}_rim-LH*layers_equivol.nii*')[0]
    depthNii = nb.load(depthFile)
    depthData = depthNii.get_fdata()
    layers = np.unique(depthData)[1:]

    # roisData = nb.load(f'{roiFolder}/sub-05_vaso_stimulation_registered_crop_largestCluster_bin_UVD_max_filter.nii.gz').get_fdata()
    # roisData = nb.load(glob.glob(f'{segFolder}/{sub}_*perimeter_chunk.nii*')[0]).get_fdata()
    roisData = nb.load(glob.glob(f'{segFolder}/{sub}_rim-LH_perimeter_chunk.nii*')[0]).get_fdata()

    roiIdx = roisData == 1

    for stimDuration in [1, 2, 4, 12, 24]:
        for modality in MODALITIES:

            stimData = nb.load(glob.glob(f'{statFolder}/{sub}_{modality}_stim_{stimDuration}s_registered_crop-toShpere*H.nii.gz')[0]).get_fdata()

            # stimData = stimData[roiIdx.astype('bool')]
            # depth = depthData[roiIdx.astype('bool')]


            # for metric, val in zip(depth,stimData):
            for layer in layers:

                layerIdx = depthData == layer
                tmp = roiIdx*layerIdx

                val = np.mean(stimData[tmp])

                subList.append(sub)
                depthList.append(layer)
                valList.append(val)
                stimList.append(stimDuration)
                modalityList.append(modality)


data = pd.DataFrame({'subject': subList, 'depth':depthList, 'value':valList, 'stim':stimList, 'modality':modalityList})


plt.style.use('dark_background')

palette = {
    'bold': 'tab:orange',
    'vaso': 'tab:blue'}


# for stimDuration in [1, 2, 4, 12, 24]:
#     fig, ax = plt.subplots()
#
#     tmp = data.loc[data['stim']==stimDuration]
#
#     sns.lineplot(data=tmp, x='depth', y='value', hue='modality', linewidth=2, palette = palette)
#
#     lim = ax.get_ylim()
#     lim[0].round()
#
#     ax.set_yticks(np.linspace(lim[0].round(), lim[1].round(),5).astype('int'))
#
#     plt.ylabel(f'z-score', fontsize=24)
#
#     # plt.title(f"{sub} {roi}-ROI {modality}", fontsize=24, pad=20)
#     plt.xlabel('WM                                CSF', fontsize=24)
#     plt.xticks([])
#     yLimits = ax.get_ylim()
#     # plt.ylim(0,yLimits[1])
#
#     plt.yticks(fontsize=18)
#
#     # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#
#     plt.legend(loc='upper left')
#
#
#     # plt.savefig(f'{root}/Group_V1_{stimType}_zScoreProfile.png', bbox_inches = "tight")
#     plt.show()


palettes = {
    'bold': ['#ff7f0e', '#ff9436', '#ffaa5e', '#ffbf86','#ffd4af'],
    'vaso': ['#1f77b4', '#2a92da', '#55a8e2', '#7fbee9', '#aad4f0']}

for modality in ['bold', 'vaso']:
    fig, ax = plt.subplots()

    tmp = data.loc[(data['modality']==modality)]

    sns.lineplot(data=tmp, x='depth', y='value', hue='stim', linewidth=2, palette = palettes[modality])

    lim = ax.get_ylim()
    lim[0].round()

    ax.set_yticks(np.linspace(lim[0].round(), lim[1].round(),5).astype('int'))

    plt.ylabel(f'z-score', fontsize=20)

    # plt.title(f"Activation across stimulus durations", fontsize=20, pad=20)
    plt.title(f"{modality}", fontsize=24, pad=20)
    plt.xlabel('WM                                          CSF', fontsize=20)
    plt.xticks([])
    yLimits = ax.get_ylim()
    # plt.ylim(0,yLimits[1])

    plt.yticks(fontsize=18)

    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='upper left')
    legend = plt.legend(title='Stim dur [s]', fontsize=14, loc = 'center left',
                           bbox_to_anchor = (1, 0.5))

    title = legend.get_title()
    title.set_fontsize(14)

    # plt.savefig(f'/Users/sebastiandresbach/Desktop/sub-all_{modality}_zScoreProfile.png', bbox_inches = "tight")
    plt.show()

for modality in ['bold', 'vaso']:
    fig, ax = plt.subplots()

    tmp = data.loc[(data['modality']==modality)&(data['subject']=='sub-06')]

    sns.lineplot(data=tmp, x='depth', y='value', hue='stim', linewidth=2, palette = palettes[modality])

    lim = ax.get_ylim()
    lim[0].round()

    ax.set_yticks(np.linspace(lim[0].round(), lim[1].round(),5).astype('int'))

    plt.ylabel(f'z-score', fontsize=20)

    # plt.title(f"Activation across stimulus durations", fontsize=20, pad=20)
    plt.title(f"{modality}", fontsize=24, pad=20)
    plt.xlabel('WM                                          CSF', fontsize=20)
    plt.xticks([])
    yLimits = ax.get_ylim()
    # plt.ylim(0,yLimits[1])

    plt.yticks(fontsize=18)

    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='upper left')
    legend = plt.legend(title='Stim dur [s]', fontsize=14, loc = 'center left',
                           bbox_to_anchor = (1, 0.5))

    title = legend.get_title()
    title.set_fontsize(14)

    # plt.savefig(f'/Users/sebastiandresbach/Desktop/sub-all_{modality}_zScoreProfile.png', bbox_inches = "tight")
    plt.show()
