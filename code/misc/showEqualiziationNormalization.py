"""Plotting the process from raw data to equalization and normalization"""
import os
import glob
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored.csv', sep=',')



plt.style.use('dark_background')

palettesLayers = {'vaso': ['#55a8e2', '#aad4f0', '#ffffff', '#FF0000'],
                  'bold': ['#ff8c26', '#ffd4af', '#ffffff', '#FF0000']}

layerNames = ['deep', 'middle', 'superficial', 'vein']


for modality in ['bold', 'vaso']:
    for stimDuration in [1., 2., 4., 12., 24.]:
        # for stimDuration in [1]:

        fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

        for layer in [1, 2, 3]:

            tmp = data.loc[(data['stimDur'] == stimDuration)
                                & (data['layer'] == layer)
                                & (data['modality'] == modality)
                                & (data['dataType'] == 'raw')
                                & (data['subject'] == 'sub-06')]

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))

            # # Get value of first volume for given layer
            # val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # # Normalize to that value
            # tmp['data'] = tmp['data'] - val

            if modality == 'vaso':
                ax1.set_ylim(-1.1, 5.1)
            if modality == 'bold':
                ax1.set_ylim(-4.1, 7.1)

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
            f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/sub-06_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers_raw.png',
            bbox_inches="tight")
        # plt.show()
        plt.close()


equalized = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalized.csv', sep=',')

for modality in ['bold', 'vaso']:
    for stimDuration in [1., 2., 4., 12., 24.]:
        # for stimDuration in [1]:

        fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

        for layer in [1, 2, 3]:

            tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                           & (equalized['layer'] == layer)
                           & (equalized['modality'] == modality)
                           & (equalized['dataType'] == 'raw')
                           & (equalized['subject'] == 'sub-06')]

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))

            # # Get value of first volume for given layer
            # val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # # Normalize to that value
            # tmp['data'] = tmp['data'] - val

            if modality == 'vaso':
                ax1.set_ylim(-1.1, 5.1)
            if modality == 'bold':
                ax1.set_ylim(-4.1, 7.1)

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
            f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/sub-06_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers_equal.png',
            bbox_inches="tight")
        # plt.show()
        plt.close()

equalizedNorm = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalizedNormalized.csv', sep=',')

for modality in ['bold', 'vaso']:
    for stimDuration in [1., 2., 4., 12., 24.]:
        # for stimDuration in [1]:

        fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

        for layer in [1, 2, 3]:

            tmp = equalizedNorm.loc[(equalizedNorm['stimDur'] == stimDuration)
                           & (equalizedNorm['layer'] == layer)
                           & (equalizedNorm['modality'] == modality)
                           & (equalizedNorm['dataType'] == 'raw')
                           & (equalizedNorm['subject'] == 'sub-06')]

            # Get number of volumes for stimulus duration
            nrVols = len(np.unique(tmp['volume']))

            # # Get value of first volume for given layer
            # val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
            # # Normalize to that value
            # tmp['data'] = tmp['data'] - val

            if modality == 'vaso':
                ax1.set_ylim(-1.1, 5.1)
            if modality == 'bold':
                ax1.set_ylim(-4.1, 7.1)

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
        if stimDuration == 24:
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
            f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs/sub-06_stimDur-{int(stimDuration):2d}_{modality}_ERA-layers_equalNorm.png',
            bbox_inches="tight")
        # plt.show()
        plt.close()