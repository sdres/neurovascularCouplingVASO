"""Plot event-related averages per stimulus duration"""

import os
import glob
import nibabel as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
# SUBS = ['sub-08']

STIMDURS = [1, 2, 4, 12, 24]

EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

MODALITIES = ['vaso', 'bold']

data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored.csv', sep=',')

equalized = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalizedNormalized.csv', sep=',')

# equalized = pd.concat((data, equalized))

# equalized = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_vaso.csv', sep=',')


# =============================================================================
# plot single subs
# =============================================================================

plt.style.use('dark_background')

palettesLayers = {'vaso': ['#55a8e2', '#aad4f0', '#ffffff', '#FF0000'],
                  'bold': ['#ff8c26', '#ffd4af', '#ffffff', '#FF0000']}

layerNames = ['deep', 'middle', 'superficial', 'vein']

for sub in SUBS:

    for modality in ['bold', 'vaso']:
    # for modality in ['vaso']:

        for stimDuration in [1., 2., 4., 12., 24.]:
            # for stimDuration in [1]:

            fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

            for layer in [1, 2, 3]:

                tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                                    & (equalized['layer'] == layer)
                                    & (equalized['modality'] == modality)
                                    & (equalized['dataType'] == 'raw')
                                    & (equalized['subject'] == sub)]

                # Get number of volumes for stimulus duration
                nrVols = len(np.unique(tmp['volume']))

                # # Get value of first volume for given layer
                # val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
                # # Normalize to that value
                # tmp['data'] = tmp['data'] - val

                if modality == 'vaso':
                    ax1.set_ylim(-1.1, 5.1)
                if modality == 'bold':
                    ax1.set_ylim(-3.1, 7.1)

                sns.lineplot(ax=ax1,
                             data=tmp,
                             x="volume",
                             y="data",
                             color=palettesLayers[modality][layer - 1],
                             linewidth=3,
                             # ci=None,
                             label=layerNames[layer - 1],
                             )

            # Set font-sizes for axes
            ax1.yaxis.set_tick_params(labelsize=18)
            ax1.xaxis.set_tick_params(labelsize=18)

            # Tweak x-axis
            ticks = np.linspace(0, nrVols, 10)
            labels = (np.linspace(0, nrVols, 10) * 0.7808410714285715).round(decimals=1)
            ax1.set_xticks(ticks[::2])
            ax1.set_xticklabels(labels[::2], fontsize=18)
            ax1.set_xlabel('Time [s]', fontsize=24)

            # Draw stimulus duration
            ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
            # Draw line at 0% signal-change
            ax1.axhline(0, linestyle='--', color='white')

            # Prepare legend
            if stimDuration == 24 and sub == 'sub-09':
                legend = ax1.legend(loc='upper right', title="Layer", fontsize=20)
                legend.get_title().set_fontsize('18')  # Legend 'Title' font-size
            else:
                ax1.get_legend().remove()

            ax1.set_ylabel(r'Signal change [%]', fontsize=24)

            # if sub == 'sub-08' and stimDuration == 1 and modality == 'vaso':
            #     for spine in ax1.spines.values():
            #         spine.set_edgecolor('red')

            plt.tight_layout()

            plt.savefig(
                f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/{sub}_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers_equalizedNormalized.png',
                bbox_inches="tight")
            # plt.show()
            plt.close()


for modality in ['bold', 'vaso']:
# for modality in ['vaso']:

    for stimDuration in [1., 2., 4., 12., 24.]:
    # for stimDuration in [1]:

        fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

        for layer in [1, 2, 3]:

            tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                                & (equalized['layer'] == layer)
                                & (equalized['modality'] == modality)
                                & (equalized['dataType'] == 'raw')
                                & (equalized['subject'] != 'sub-08')]

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))

            # # Get value of first volume for given layer
            # val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # # Normalize to that value
            # tmp['data'] = tmp['data'] - val

            if modality == 'vaso':
                ax1.set_ylim(-1.1, 5.1)
            if modality == 'bold':
                ax1.set_ylim(-3.1, 7.1)

            sns.lineplot(ax=ax1,
                         data=tmp,
                         x="volume",
                         y="data",
                         color=palettesLayers[modality][layer-1],
                         linewidth=3,
                         # ci=None,
                         label=layerNames[layer-1],
                         )

        # Set font-sizes for axes
        ax1.yaxis.set_tick_params(labelsize=18)
        ax1.xaxis.set_tick_params(labelsize=18)

        # Tweak x-axis
        ticks = np.linspace(0, nrVols, 10)
        labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)
        ax1.set_xticks(ticks[::2])
        ax1.set_xticklabels(labels[::2], fontsize=18)
        ax1.set_xlabel('Time [s]', fontsize=24)

        # Draw stimulus duration
        ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
        # Draw line at 0% signal-change
        ax1.axhline(0, linestyle='--', color='white')

        # Prepare legend
        if stimDuration == 24:
            legend = ax1.legend(loc='upper right', title="Layer", fontsize=18)
            legend.get_title().set_fontsize('18')  # Legend 'Title' font-size
        else:
            ax1.get_legend().remove()

        ax1.set_ylabel(r'Signal change [%]', fontsize=24)

        # # Set title
        # titlePad = 10
        # if stimDuration == 1:
        #     plt.title(f'{int(stimDuration)} second stimulation', fontsize=24, pad=titlePad)
        # else:
        #     plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24, pad=titlePad)

        plt.tight_layout()

        plt.savefig(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/group_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers.png',
                    bbox_inches="tight")
        plt.show()

# =============================================================================
# plot 2s stimulation for review of event-related
# =============================================================================

for modality in ['bold', 'vaso']:
# for modality in ['bold']:

    # for stimDuration in [1., 2., 4., 12., 24.]:
    for stimDuration in [2]:

        fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

        for layer in [1, 2, 3]:

            tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                                & (equalized['layer'] == layer)
                                & (equalized['modality'] == modality)
                                & (equalized['dataType'] == 'raw')]

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))

            # Get value of first volume for given layer
            val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # Normalize to that value
            tmp['data'] = tmp['data'] - val

            if modality == 'vaso':
                ax1.set_ylim(-1.1, 3.1)
            if modality == 'bold':
                ax1.set_ylim(-1.1, 4.1)

            sns.lineplot(ax=ax1,
                         data=tmp,
                         x="volume",
                         y="data",
                         color=palettesLayers[modality][layer-1],
                         linewidth=3,
                         # ci=None,
                         label=layerNames[layer-1],
                         )

        # Set font-sizes for axes
        ax1.yaxis.set_tick_params(labelsize=18)
        ax1.xaxis.set_tick_params(labelsize=18)

        # Tweak x-axis
        ticks = np.linspace(0, nrVols, 10)
        labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)
        ax1.set_xticks(ticks[::2])
        ax1.set_xticklabels(labels[::2], fontsize=18)
        ax1.set_xlabel('Time [s]', fontsize=24)

        # Draw stimulus duration
        ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
        # Draw line at 0% signal-change
        ax1.axhline(0, linestyle='--', color='white')

        # Prepare legend
        legend = ax1.legend(loc='upper right', title="Layer", fontsize=18)
        legend.get_title().set_fontsize('18')  # Legend 'Title' font-size

        ax1.set_ylabel(r'Signal change [%]', fontsize=24)

        # # Set title
        # titlePad = 10
        # if stimDuration == 1:
        #     plt.title(f'{int(stimDuration)} second stimulation', fontsize=24, pad=titlePad)
        # else:
        #     plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24, pad=titlePad)

        plt.tight_layout()

        plt.savefig(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/group_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers_review.png',
                    bbox_inches="tight")
        plt.show()

# =============================================================================
# plot zscored ERAs
# =============================================================================
equalized = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalizedNormalized.csv', sep=',')

plt.style.use('dark_background')

palettesLayers = {'vaso': ['#55a8e2', '#aad4f0', '#ffffff', '#FF0000'],
                  'bold': ['#ff8c26', '#ffd4af', '#ffffff', '#FF0000']}

layerNames = ['deep', 'middle', 'superficial', 'vein']

for sub in ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']:
# for sub in ['sub-08']:

    # for modality in ['bold', 'vaso']:
    for modality in ['bold']:

        for stimDuration in [1., 2., 4., 12., 24.]:
        # for stimDuration in [1]:

            fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

            for layer in [1, 2, 3]:

                tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                               & (equalized['layer'] == layer)
                               & (equalized['modality'] == modality)
                               & (equalized['subject'] == sub)
                               & (equalized['dataType'] == 'zscore')]

                # Get number of volumes for stimulus duration
                nrVols = len(np.unique(tmp['volume']))

                # Get value of first volume for given layer
                # val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
                # Normalize to that value
                # tmp['data'] = tmp['data'] - val

                if modality == 'vaso':
                    ax1.set_ylim(-1.1, 1.6)

                if modality == 'bold':
                    ax1.set_ylim(-1.6, 2.1)

                sns.lineplot(ax=ax1,
                             data=tmp,
                             x="volume",
                             y="data",
                             color=palettesLayers[modality][layer-1],
                             linewidth=3,
                             # ci=None,
                             label=layerNames[layer-1],
                             )

            # Set font-sizes for axes
            ax1.yaxis.set_tick_params(labelsize=18)
            ax1.xaxis.set_tick_params(labelsize=18)

            # Tweak x-axis
            ticks = np.linspace(0, nrVols, 10)
            labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)
            ax1.set_xticks(ticks[::2])
            ax1.set_xticklabels(labels[::2], fontsize=18)
            ax1.set_xlabel('Time [s]', fontsize=24)

            # Draw stimulus duration
            ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')

            # Prepare legend
            if stimDuration == 24:
                legend = ax1.legend(loc='upper right', title="Layer", fontsize=18)
                legend.get_title().set_fontsize('18')  # Legend 'Title' font-size
            else:
                ax1.get_legend().remove()

            ax1.set_ylabel(r'Sig. change [%, z-scored]', fontsize=24)

            plt.tight_layout()
            plt.savefig(f'./results/ERAs/{sub}_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers_zscored.png',
                        bbox_inches="tight")
            plt.show()

for modality in ['bold', 'vaso']:
# for modality in ['bold']:

    for stimDuration in [1., 2., 4., 12., 24.]:

        fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

        for layer in [1, 2, 3]:

            tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                                & (equalized['layer'] == layer)
                                & (equalized['modality'] == modality)
                                & (equalized['dataType'] == 'zscore')
                                & (equalized['subject'] != 'sub-08')]

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))

            # # Get value of first volume for given layer
            # val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # # Normalize to that value
            # tmp['data'] -= val

            if modality == 'vaso':
                ax1.set_ylim(-1.1, 1.6)

            if modality == 'bold':
                ax1.set_ylim(-1.6, 2.1)

            sns.lineplot(ax=ax1,
                         data=tmp,
                         x="volume",
                         y="data",
                         color=palettesLayers[modality][layer-1],
                         linewidth=3,
                         # ci=None,
                         label=layerNames[layer-1],
                         )

        # Set font-sizes for axes
        ax1.yaxis.set_tick_params(labelsize=18)
        ax1.xaxis.set_tick_params(labelsize=18)

        # Tweak x-axis
        ticks = np.linspace(0, nrVols, 10)
        labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)
        ax1.set_xticks(ticks[::2])
        ax1.set_xticklabels(labels[::2], fontsize=18)
        ax1.set_xlabel('Time [s]', fontsize=24)

        # Draw stimulus duration
        ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
        # Draw line at 0% signal-change
        # ax1.axhline(0, linestyle='--', color='white')

        # Prepare legend
        if stimDuration == 24:
            legend = ax1.legend(loc='upper right', title="Layer", fontsize=18)
            legend.get_title().set_fontsize('18')  # Legend 'Title' font-size
        else:
            ax1.get_legend().remove()

        ax1.set_ylabel(r'Sig. change [%, z-scored]', fontsize=24)

        plt.tight_layout()
        plt.savefig(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/group_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers_zscored.png',
                    bbox_inches="tight")
        plt.show()


# =====================================================================================================================
# Plot ERA per layer
# =====================================================================================================================

# Define figzize
FS = (8, 5)
# define linewidth to 2
LW = 2
# Define fontsize size for x- and y-labels
labelSize = 24
# Define fontsize size for x- and y-ticks
tickLabelSize = 18
# Define fontsize legend text
legendTextSize = 18
palettes = {
    'bold': ['#ff7f0e', '#ff9436', '#ffaa5e', '#ffbf86', '#ffd4af'],
    'vaso': ['#1f77b4', '#2a92da', '#55a8e2', '#7fbee9', '#aad4f0']}

for j, modality in enumerate(['bold', 'vaso']):
    fig, axes = plt.subplots(1, 3, figsize=(21, 5), sharey=True)
    fig.subplots_adjust(top=0.8)

    for i, layer in enumerate(['deep', 'middle', 'superficial']):

        for k, stimDur in enumerate(data['stimDur'].unique()):
            # tmp = data.loc[(data['modality'] == modality)
            #                & (data['layer'] == i+1)
            #                & (data['stimDur'] == stimDur)
            #                & (data['dataType'] == 'raw')]
            tmp = equalized.loc[(equalized['modality'] == modality)
                           & (equalized['layer'] == i+1)
                           & (equalized['stimDur'] == stimDur)
                           & (equalized['dataType'] == 'raw')]


            # # Get value of first volume for given layer
            val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # # Normalize to that value
            tmp['data'] = tmp['data'] - val

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))
            sns.lineplot(ax=axes[i],
                         data=tmp,
                         x='volume',
                         y='data',
                         hue='stimDur',
                         linewidth=LW,
                         palette=[palettes[modality][k]])
                         # errorbar=None)

        # ================================================================================
        # Tweak x-axis
        ticks = np.linspace(0, nrVols, 6)
        labels = (np.linspace(0, nrVols, 6) * 0.7808410714285715).astype('int')
        axes[i].set_xticks(ticks)
        axes[i].set_xticklabels(labels, fontsize=18)
        if i == 1:
            axes[i].set_xlabel('Time [s]', fontsize=24)
        else:
            axes[i].set_xlabel('', fontsize=24)

        # ================================================================================
        # Tweak y-axis
        if modality == 'vaso':
            axes[i].set_ylim(-1.1, 4.1)
        if modality == 'bold':
            axes[i].set_ylim(-3.3, 6.1)

        lim = axes[0].get_ylim()
        tickMarks = np.arange(lim[0].round(), lim[1], 1).astype('int')

        if i == 0:
            axes[i].set_ylabel(r'Signal change [%]', fontsize=labelSize)
            axes[i].set_yticks(tickMarks, tickMarks, fontsize=18)
        # elif i > 1:
        #     axes[i].set_ylabel(r'', fontsize=labelSize)
        #     axes[i].set_yticks([])

        # Set font-sizes for axes
        axes[i].yaxis.set_tick_params(labelsize=18)
        axes[i].xaxis.set_tick_params(labelsize=18)

        # ================================================================================
        # Misc
        axes[i].set_title(layer, fontsize=labelSize)
        # Draw lines
        axes[i].axhline(0, linestyle='--', color='white')
        # Legend
        if i < 3:
            axes[i].get_legend().remove()

        legend = axes[2].legend(title='Stim dur [s]', loc='upper right', fontsize=14)
        title = legend.get_title()
        title.set_fontsize(18)

    # plt.suptitle(f'{modality}', fontsize=labelSize, y=0.98)
    plt.tight_layout()
    plt.savefig(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/group_byLayer_{modality}_ERA.png',
                bbox_inches="tight")
    plt.show()


# Zoomed in
for j, modality in enumerate(['bold', 'vaso']):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.subplots_adjust(top=0.8)

    for i, layer in enumerate(['deep', 'middle', 'superficial']):

        for k, stimDur in enumerate(data['stimDur'].unique()):
            tmp = data.loc[(data['modality'] == modality)
                           & (data['layer'] == i + 1)
                           & (data['stimDur'] == stimDur)
                           & (data['dataType'] == 'raw')
                           & (data['volume'] <= 6)]

            # # Get value of first volume for given layer
            val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # # Normalize to that value
            tmp['data'] = tmp['data'] - val

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))
            sns.lineplot(ax=axes[i],
                         data=tmp,
                         x='volume',
                         y='data',
                         hue='stimDur',
                         linewidth=LW,
                         palette=[palettes[modality][k]],
                         errorbar=None)
        #
        # # ================================================================================
        # # Tweak x-axis
        # ticks = np.linspace(0, nrVols, 6)
        # labels = (np.linspace(0, nrVols, 6) * 0.7808410714285715).astype('int')
        # axes[i].set_xticks(ticks)
        # axes[i].set_xticklabels(labels, fontsize=18)
        # if i == 1:
        #     axes[i].set_xlabel('Time [s]', fontsize=24)
        # else:
        #     axes[i].set_xlabel('', fontsize=24)
        #
        # # ================================================================================
        # # Tweak y-axis
        # if modality == 'vaso':
        #     axes[i].set_ylim(-1.1, 4.1)
        # if modality == 'bold':
        #     axes[i].set_ylim(-3.3, 6.1)
        #
        # lim = axes[0].get_ylim()
        # tickMarks = np.arange(lim[0].round(), lim[1], 1).astype('int')
        #
        # if i == 0:
        #     axes[i].set_ylabel(r'Signal change [%]', fontsize=labelSize)
        #     axes[i].set_yticks(tickMarks, tickMarks, fontsize=18)
        # # elif i > 1:
        # #     axes[i].set_ylabel(r'', fontsize=labelSize)
        # #     axes[i].set_yticks([])
        #
        # # Set font-sizes for axes
        # axes[i].yaxis.set_tick_params(labelsize=18)
        # axes[i].xaxis.set_tick_params(labelsize=18)
        #
        # # ================================================================================
        # # Misc
        # axes[i].set_title(layer, fontsize=labelSize)
        # # Draw lines
        # axes[i].axhline(0, linestyle='--', color='white')
        # # Legend
        # if i < 3:
        #     axes[i].get_legend().remove()
        # legend = axes[2].legend(title='Stim dur [s]', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        # title = legend.get_title()
        # title.set_fontsize(18)

    # plt.suptitle(f'{modality}', fontsize=labelSize, y=0.98)
    plt.tight_layout()
    # plt.savefig(
    #     f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/group_byLayer_{modality}_ERA.png',
    #     bbox_inches="tight")
    plt.show()