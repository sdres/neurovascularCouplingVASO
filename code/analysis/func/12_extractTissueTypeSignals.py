"""

Extracting and plotting signal based on ME-GRE vessel segmentations

"""

import glob
import nibabel as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
import seaborn as sns
import sys
sys.path.append('./code/misc')
from findTr import *
plt.style.use('dark_background')

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'
# Define subjects to work on
SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-09']

# Get TR
UPFACTOR = 4

# Get propper ROIs for GM and vessels

for sub in SUBS:
    subDir = f'{DATADIR}/{sub}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    # Find MEGRE session
    for sesNr in range(1, 6):
        if os.path.exists(f"{DATADIR}/{sub}/ses-0{sesNr}/anat/megre/11_T2star/"
                          f"{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz"):
            megreSes = f'ses-0{sesNr}'

    # Load vessel file
    vesselFile = f'{DATADIR}/{sub}/{megreSes}/anat/megre/12_vessels/{sub}_vessels_reg_crop-toSphereLH.nii.gz'
    vesselNii = nb.load(vesselFile)
    vesselData = vesselNii.get_fdata()

    # Dilate vessels
    struct = generate_binary_structure(3, 1)  # 1 jump neighbourbhood
    vesselDataDilated = scipy.ndimage.binary_dilation(vesselData, structure=struct, iterations=1)

    # Save for QA
    img = nb.Nifti1Image(vesselDataDilated, affine=vesselNii.affine, header=vesselNii.header)
    outName = f'{vesselFile.split(".")[0]}_dilate.nii.gz'
    nb.save(img, outName)

    # np.unique(vesselDataDilated)
    # # Multiply vessels to differentiate from GM
    # vesselDataDilated = vesselDataDilated * 2
    #
    # # # Load layer file
    # # gmData = nb.load(f'{segFolder}/3layers_layers_equivol.nii').get_fdata()
    # # # Binarize to get GM
    # # np.unique(gmData)
    # #
    # # gmData = np.where(gmData > 0 , 1, 0)
    #
    # # # Add tissue types
    # # vesselGM = vesselDataDilated + gmData
    # # # Now, we have an array where GM == 1, Pial vessels == 2, intracortical vessels == 3
    # # # However, we want to limit ourselves to the perimeter chunk, so all the GM has to go
    # # vessels = (vesselGM > 1) * vesselGM
    # # # Save for QA
    # # img = nb.Nifti1Image(vessels, affine = vesselNii.affine, header = vesselNii.header)
    # # nb.save(img, f'{DATADIR}/{sub}/ses-04/anat/megre/vesselsmulti.nii.gz')
    #
    # roisData = nb.load(f'{segFolder}/{sub}_rim-LH_perimeter_chunk.nii.gz').get_fdata()
    # roisData = np.where(roisData == 1, 1, 0)
    #
    # # Before adding our ROI-GM to the vessels, we have to mask out vessel voxels
    # # inverseVessels = np.where(vessels >= 1, 0, 1)
    #
    # # roisData = np.multiply(roisData, inverseVessels)
    #
    # # vesselGM = vessels + roisData
    # vesselGM = vesselDataDilated + roisData
    # # Now, we have an array where ROI-GM == 1, Pial vessels == 2, intracortical vessels == 3
    # vesselGM = np.where(vesselGM == 3, 2, vesselGM)
    # # Save for QA
    # img = nb.Nifti1Image(vesselGM, affine=vesselNii.affine, header=vesselNii.header)
    # nb.save(img, f'{DATADIR}/{sub}/{megreSes}/anat/megre/vesselsPlusPerimeter.nii.gz')
    #


# =============================================================================
# extract from vessels
# =============================================================================

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []
tissueList = []

tissues = {1: 'Gray matter', 2: 'Vessel dominated'}

for sub in SUBS:
    subDir = f'{DATADIR}/{sub}'

    # Find MEGRE session
    for sesNr in range(1, 6):
        if os.path.exists(f"{DATADIR}/{sub}/ses-0{sesNr}/anat/megre/11_T2star/"
                          f"{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz"):
            megreSes = f'ses-0{sesNr}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    vesselFile = f'{DATADIR}/{sub}/{megreSes}/anat/megre/12_vessels/{sub}_vessels_reg_crop-toSphereLH.nii.gz'
    vesselNii = nb.load(vesselFile)
    vesselData = vesselNii.get_fdata()

    roisData = nb.load(f'{segFolder}/{sub}_rim-LH_perimeter_chunk.nii.gz').get_fdata()
    roisData = np.where(roisData == 1, 1, 0)

    # Make single mask array
    vesselData *= 2

    vesselGM = roisData + vesselData

    for stimDuration in [1, 2, 4, 12, 24]:

        for modality in ['vaso', 'bold']:
            frames = sorted(glob.glob(f'{DATADIR}/{sub}/ERAs/frames/{sub}_ses-avg_task-stimulation_run-avg_part-mag'
                                      f'_{modality}_intemp_era-{stimDuration}s_sigChange-after_frame*_registered_crop.nii.gz'))

            for j, frame in enumerate(frames):

                nii = nb.load(frame)
                data = nii.get_fdata()
                data.shape
                for key in tissues.keys():

                    idx = vesselGM == key

                    val = np.mean(data[idx])

                    valList.append(val)
                    subList.append(sub)
                    tissueList.append(tissues[key])
                    stimDurList.append(stimDuration)
                    modalityList.append(modality)
                    timePointList.append(j)

data = pd.DataFrame({'subject': subList,
                     'volume': timePointList,
                     'modality': modalityList,
                     'data': valList,
                     'tissue': tissueList,
                     'stimDur': stimDurList
                     })

# =============================================================================
# Equalize mean between first and second set
# =============================================================================

# Equalize mean between first and second set
tr = 3.1262367768477795/4
EVENTDURS = {'shortITI': (np.array([11, 14, 18, 32, 48])/tr).astype('int'),
             'longITI': (np.array([21, 24, 28, 42, 64])/tr).astype('int')}

STIMDURS = [1, 2, 4, 12, 24]

equalized = pd.DataFrame()

for sub in data['subject'].unique():
    for modality in ['vaso', 'bold']:
    # for modality in ['vaso']:
            for layer in data['tissue'].unique():
            for i, stimDur in enumerate(STIMDURS):
                tmp = data.loc[(data['subject'] == sub)
                               & (data['modality'] == modality)
                               & (data['tissue'] == layer)
                               & (data['stimDur'] == stimDur)]

                extension = EVENTDURS['longITI'][i] - EVENTDURS['shortITI'][i]
                # Get max number of volumes
                maxVol = np.max(tmp['volume'].to_numpy())

                firstVol = maxVol - extension

                series1 = tmp.loc[(tmp['volume'] < firstVol+1)]
                series2 = tmp.loc[(tmp['volume'] >= firstVol+1)]

                val1 = np.mean(series1.loc[series1['volume'] == firstVol]['data'])
                val2 = np.mean(series2.loc[series2['volume'] == firstVol+1]['data'])

                diff = val1 - val2
                series2['data'] += diff

                equalized = pd.concat((equalized, series1))
                equalized = pd.concat((equalized, series2))

equalized.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_vessels_equalized.csv',
            sep=',',
            index=False)


equalized = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_vessels_equalized.csv', sep=',')

normalized = pd.DataFrame()

for sub in SUBS:
    for modality in ['bold', 'vaso']:
        for stimDuration in [1., 2., 4., 12., 24.]:
            for layer in equalized['tissue'].unique():

                tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                                    & (equalized['tissue'] == layer)
                                    & (equalized['modality'] == modality)
                                    & (equalized['subject'] == sub)]


                # Get value of first volume for given layer
                val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
                # Normalize to that value
                tmp['data'] = tmp['data'] - val

                normalized = pd.concat((normalized, tmp))

normalized.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_vessels_equalizedNormalized.csv',
            sep=',',
            index=False)


# =============================================================================
# Plotting
# =============================================================================

palettesLayers = {'vaso': ['#55a8e2', '#FF0000'], 'bold': ['#ff8c26', '#FF0000']}

for sub in normalized['subject'].unique():
    for modality in ['vaso', 'bold']:

        for stimDuration in [1., 2., 4., 12., 24.]:
            fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

            for j, tissue in enumerate(['Gray matter', 'Vessel dominated']):

                tmp = normalized.loc[(normalized['stimDur'] == stimDuration)
                               & (normalized['tissue'] == tissue)
                               & (normalized['modality'] == modality)
                               & (normalized['subject'] == sub)]


                nrVols = len(np.unique(tmp['volume']))

                sns.lineplot(ax=ax1,
                             data=tmp,
                             x="volume",
                             y="data",
                             color=palettesLayers[modality][j],
                             linewidth=3,
                             label=tissue,
                             )

            if modality == 'vaso':
                ax1.set_ylim(-2, 4)
                yTickVals = np.arange(-2, 4.1, 1)
                ax1.set_yticks(yTickVals)

            if modality == 'bold':
                ax1.set_ylim(-4, 15)
                yTickVals = np.arange(-3, 15.1, 3)
                ax1.set_yticks(yTickVals)

            # Prepare x-ticks
            ticks = np.linspace(0, nrVols, 10)
            labels = (ticks * 0.7808410714285715).round(decimals=1)

            ax1.yaxis.set_tick_params(labelsize=18)
            ax1.xaxis.set_tick_params(labelsize=18)

            # Tweak x-axis
            ax1.set_xticks(ticks[::2])
            ax1.set_xticklabels(labels[::2], fontsize=18)
            ax1.set_xlabel('Time [s]', fontsize=24)

            # Draw lines for stim duration and 0-line
            ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
            ax1.axhline(0, linestyle='--', color='white')

            # Prepare legend
            if stimDuration == 24 and sub == 'sub-09':
                legend = ax1.legend(loc='upper right', title="Layer", fontsize=20)
                legend.get_title().set_fontsize('18')  # Legend 'Title' font-size
            else:
                ax1.get_legend().remove()


            ax1.set_ylabel(r'Signal change [%]', fontsize=24)

            plt.tight_layout()
            plt.savefig(f'./results/ERAs/{sub}_stimDur-{int(stimDuration)}_{modality}_ERA-tissues.png', bbox_inches="tight")
            plt.close()



palettesLayers = {'vaso': ['#55a8e2', '#FF0000'], 'bold': ['#ff8c26', '#FF0000']}

for modality in ['vaso', 'bold']:

    for stimDuration in [1., 2., 4., 12., 24.]:
        fig, (ax1) = plt.subplots(1, 1, figsize=(7.5, 5))

        for j, tissue in enumerate(['Gray matter', 'Vessel dominated']):

            tmp = normalized.loc[(normalized['stimDur'] == stimDuration)
                           & (normalized['tissue'] == tissue)
                           & (normalized['modality'] == modality)]


            nrVols = len(np.unique(tmp['volume']))

            sns.lineplot(ax=ax1,
                         data=tmp,
                         x="volume",
                         y="data",
                         color=palettesLayers[modality][j],
                         linewidth=3,
                         label=tissue,
                         )

        if modality == 'vaso':
            ax1.set_ylim(-2, 4)
            yTickVals = np.arange(-2, 4.1, 1)
            ax1.set_yticks(yTickVals)

        if modality == 'bold':
            ax1.set_ylim(-4, 14)
            yTickVals = np.arange(-3, 12.1, 3)
            ax1.set_yticks(yTickVals)

        # Prepare x-ticks
        ticks = np.linspace(0, nrVols, 10)
        labels = (ticks * 0.7808410714285715).round(decimals=1)

        ax1.yaxis.set_tick_params(labelsize=18)
        ax1.xaxis.set_tick_params(labelsize=18)

        # Tweak x-axis
        ax1.set_xticks(ticks[::2])
        ax1.set_xticklabels(labels[::2], fontsize=18)
        ax1.set_xlabel('Time [s]', fontsize=24)

        # Draw lines for stim duration and 0-line
        ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation')
        ax1.axhline(0, linestyle='--', color='white')

        # Prepare legend
        if stimDuration == 24:
            legend = ax1.legend(loc='upper right', title="Layer", fontsize=20)
            legend.get_title().set_fontsize('18')  # Legend 'Title' font-size
        else:
            ax1.get_legend().remove()

        ax1.set_ylabel(r'Signal change [%]', fontsize=24)

        plt.tight_layout()
        plt.savefig(f'./results/ERAs/group_stimDur-{int(stimDuration)}_{modality}_ERA-tissues.png', bbox_inches="tight")
        plt.close()

# =============================================================================
# GM/Vessel ratio
# =============================================================================

# Load data
data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_vessels_equalizedNormalized.csv', sep=',')

layerList = []
valList = []
stimDurList = []
timePointList = []
subList = []
modalityList = []

for sub in data['subject'].unique():
    for modality in ['vaso', 'bold']:
        for stimDur in data['stimDur'].unique():
            for layer in data['tissue'].unique():

                tmp = data.loc[(data['subject'] == sub)
                               & (data['modality'] == modality)
                               & (data['tissue'] == layer)
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

                layerList.append(layer)
                valList.append(maxVal)
                stimDurList.append(stimDur)
                timePointList.append(maxTP)
                subList.append(sub)
                modalityList.append(modality)

peakTimeList = [i * 0.785 for i in timePointList]

timepointData = pd.DataFrame({'subject': subList,
                              'maxVol': timePointList,
                              'data': valList,
                              'layer': layerList,
                              'stimDur': stimDurList,
                              'peakTime': peakTimeList,
                              'modality': modalityList})




for modality in ['bold']:

    fig, axes = plt.subplots(1, 5, figsize=(21, 5), sharey=True)

    for i, stimDur in enumerate(temporalData['stimDur'].unique()):

        tmp = temporalData.loc[(timepointData['stimDur'] == stimDur)
                               & (timepointData['modality'] == modality)]

        sns.barplot(ax=axes[i], data=tmp, x="stimDur", y='layer', hue="layer", palette=palettesLayers[modality])

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
    # plt.savefig(
    #     f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/PSU_{modality}_{dataType}.png',
    #     bbox_inches="tight")
    plt.show()
