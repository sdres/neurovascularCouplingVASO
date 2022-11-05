'''

Extracting and plotting signal based on ME-GRE vessel segmentations

'''
import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import morphology, generate_binary_structure
import seaborn as sns
import sys
sys.path.append('./code/misc')
from findTr import *
plt.style.use('dark_background')

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'
# Define subjects to work on
SUBS = ['sub-06']

# Get TR
UPFACTOR = 4


# Get propper ROIs for GM and vessels
for sub in SUBS:
    subDir = f'{DATADIR}/{sub}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'
    # Load vessel file
    vesselFile = f'{DATADIR}/{sub}/ses-04/anat/megre/finalVeins_registered_crop.nii.gz'
    vesselNii = nb.load(vesselFile)
    vesselData = vesselNii.get_fdata()
    # Dilate vessels
    struct = generate_binary_structure(3, 1)  # 1 jump neighbourbhood
    vesselDataDilated = morphology.binary_dilation(vesselData, structure=struct, iterations=1)

    # Save for QA
    img = nb.Nifti1Image(vesselDataDilated, affine = vesselNii.affine, header = vesselNii.header)
    nb.save(img, f'{DATADIR}/{sub}/ses-04/anat/megre/finalVeins_registered_crop_dilate.nii.gz')

    np.unique(vesselDataDilated)
    # Multiply vessels to differentiate from GM
    vesselDataDilated = vesselDataDilated *2

    # # Load layer file
    # gmData = nb.load(f'{segFolder}/3layers_layers_equivol.nii').get_fdata()
    # # Binarize to get GM
    # np.unique(gmData)
    #
    # gmData = np.where(gmData > 0 , 1, 0)
    #
    # np.unique(gmData)
    #
    # # Add tissue types
    # vesselGM = vesselDataDilated + gmData
    # # Now, we have an array where GM == 1, Pial vessels == 2, intracortical vessels == 3
    # # However, we want to limit ourselves to the perimeter chunk, so all the GM has to go
    # vessels = (vesselGM > 1) * vesselGM
    # # Save for QA
    # img = nb.Nifti1Image(vessels, affine = vesselNii.affine, header = vesselNii.header)
    # nb.save(img, f'{DATADIR}/{sub}/ses-04/anat/megre/vesselsmulti.nii.gz')

    roisData = nb.load(f'{segFolder}/{sub}_rim_perimeter_chunk.nii.gz').get_fdata()
    roisData = np.where(roisData == 1, 1, 0)

    # Before adding our ROI-GM to the vessels, we have to mask out vessel voxels
    # inverseVessels = np.where(vessels >= 1, 0, 1)

    # roisData = np.multiply(roisData, inverseVessels)

    # vesselGM = vessels + roisData
    vesselGM = vesselDataDilated + roisData
    # Now, we have an array where ROI-GM == 1, Pial vessels == 2, intracortical vessels == 3
    vesselGM = np.where(vesselGM == 3, 2, vesselGM)
    # Save for QA
    img = nb.Nifti1Image(vesselGM, affine = vesselNii.affine, header = vesselNii.header)
    nb.save(img, f'{DATADIR}/{sub}/ses-04/anat/megre/vesselsPlusPerimeter.nii.gz')

np.unique(vesselGM)

SUBS = ['sub-06']

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []
depthList = []

for sub in SUBS:
    subDir = f'{DATADIR}/{sub}'
    #
    # segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'
    #
    # depthFile = f'{segFolder}/3layers_layers_equivol.nii'
    # depthNii = nb.load(depthFile)
    # depthData = depthNii.get_fdata()
    # layers = np.unique(depthData)[1:]
    #
    # # roisData = nb.load(f'{roiFolder}/sub-05_vaso_stimulation_registered_crop_largestCluster_bin_UVD_max_filter.nii.gz').get_fdata()
    # roisData = nb.load(f'{segFolder}/{sub}_rim_perimeter_chunk.nii.gz').get_fdata()
    # roiIdx = roisData == 1


    for stimDuration in [1, 2, 4, 12, 24]:
    # for stimDuration in [2]:

        for modality in ['vaso', 'bold']:
        # for modality in ['bold']:
            frames = sorted(glob.glob(f'{DATADIR}/{sub}/ERAs/frames/{sub}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDuration}s_sigChange_masked_frame*_registered_crop.nii.gz'))
            # file = f'{DATADIR}/{sub}/ERAs/frames/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-12s_sigChange-after_frame12_registered_crop.nii.gz'

            for j,frame in enumerate(frames):

                nii = nb.load(frame)

                data = nii.get_fdata()

                for layer in layers:

                    layerIdx = depthData == layer
                    tmp = roiIdx*layerIdx

                    val = np.mean(data[tmp])

                    if modality == 'bold':
                        valList.append(val)
                    if modality == 'vaso':
                        valList.append(val)

                    subList.append(sub)
                    depthList.append(layer)
                    stimDurList.append(stimDuration)
                    modalityList.append(modality)
                    timePointList.append(j)

data = pd.DataFrame({'subject': subList, 'volume': timePointList, 'modality': modalityList, 'data': valList, 'layer':depthList, 'stimDur':stimDurList})





# =============================================================================
# extract from vessels
# =============================================================================

SUBS = ['sub-06']

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []
tissueList = []

tissues = {1:'Gray matter', 2:'Vessel dominated', 3: 'intracortical vessel'}

for sub in SUBS:
    subDir = f'{DATADIR}/{sub}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    # vesselFile = f'/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/finalVeins_registered_crop.nii.gz'
    # vesselNii = nb.load(vesselFile)
    # vesselData = vesselNii.get_fdata()
    # vesseTypes = np.unique(vesselData)[1:]
    #
    # # roisData = nb.load(f'{roiFolder}/sub-05_vaso_stimulation_registered_crop_largestCluster_bin_UVD_max_filter.nii.gz').get_fdata()
    # roisData = nb.load(f'{segFolder}/{sub}_rim_perimeter_chunk.nii.gz').get_fdata()
    # roiIdx = roisData == 1
    tissueTypes = np.unique(vesselGM)[1:]

    for stimDuration in [1, 2, 4, 12, 24]:
    # for stimDuration in [2]:

        for modality in ['vaso', 'bold']:
        # for modality in ['bold']:
            frames = sorted(glob.glob(f'{DATADIR}/{sub}/ERAs/frames/{sub}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDuration}s_sigChange_masked_frame*_registered_crop.nii.gz'))
            # file = f'{DATADIR}/{sub}/ERAs/frames/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-12s_sigChange-after_frame12_registered_crop.nii.gz'

            for j,frame in enumerate(frames):

                nii = nb.load(frame)

                data = nii.get_fdata()

                for tissue in tissueTypes:


                    idx = vesselGM == tissue
                    # tmp = roiIdx*layerIdx

                    val = np.mean(data[idx])

                    if modality == 'bold':
                        valList.append(val)
                    if modality == 'vaso':
                        valList.append(val)

                    subList.append(sub)


                    tissueList.append(tissues[tissue])
                    stimDurList.append(stimDuration)
                    modalityList.append(modality)
                    timePointList.append(j)

data = pd.DataFrame({'subject': subList, 'volume': timePointList, 'modality': modalityList, 'data': valList, 'tissue':tissueList, 'stimDur':stimDurList})





palettesLayers = {'vaso':['#55a8e2','#FF0000'],
'bold':['#ff8c26', '#FF0000']}
# tissues = {1:'GM', 2:'Pial Vessel', 3: 'intracortical vessel'}

# for interpolationType in ['linear', 'cubic']:
for interpolationType in ['linear']:
    # data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/{sub}_task-stimulation_responses.csv', sep = ',')
    # data = data.loc[data['interpolation']==interpolationType]
    for modality in ['bold', 'vaso']:
    # for modality in ['vaso']:

        for stimDuration in [1., 2., 4., 12., 24.]:
            fig, (ax1) = plt.subplots(1,1,figsize=(7.5,5))

            # for modality in ['bold', 'vaso']:

            for j, tissue in enumerate(['Gray matter','Vessel dominated']):

                # val = np.mean(data.loc[(data['stimDur'] == stimDuration)
                #                      & (data['volume'] == 0)
                #                      & (data['layer'] == layer)]['data']
                #                      )


                tmp = data.loc[(data['stimDur'] == stimDuration)&(data['tissue'] == tissue)&(data['modality'] == modality)&(data['subject'] == 'sub-06')]

                # if val > 0:
                #     tmp['data'] = tmp['data'] - val
                # if val < 0:
                    # tmp['data'] = tmp['data'] + val
                # tmp['data'] = tmp['data'] - val
                nrVols = len(np.unique(tmp['volume']))

                # ax1.set_xticks(np.arange(-1.5,3.6))
                if modality == 'vaso':
                    ax1.set_ylim(-5.1,7.1)
                if modality == 'bold':
                    ax1.set_ylim(-8.1,12.1)
                sns.lineplot(ax=ax1,
                             data = tmp,
                             x = "volume",
                             y = "data",
                             color = palettesLayers[modality][j],
                             linewidth = 3,
                             # ci=None,
                             label = tissue,
                             )
            if modality == 'vaso':
                ax1.set_ylim(-5.1,7.1)
            if modality == 'bold':
                ax1.set_ylim(-8.1,12.1)
            # Prepare x-ticks
            ticks = np.linspace(0, nrVols, 10)
            labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)

            # ax1.set_yticks(np.arange(-0.25, 3.51, 0.5))

            ax1.yaxis.set_tick_params(labelsize=18)
            ax1.xaxis.set_tick_params(labelsize=18)

            # tweak x-axis
            ax1.set_xticks(ticks[::2])
            ax1.set_xticklabels(labels[::2],fontsize=18)
            ax1.set_xlabel('Time [s]', fontsize=24)

            # draw lines
            ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation')
            # get value of first timepoint

            ax1.axhline(0,linestyle = '--', color = 'white')

            legend = ax1.legend(loc='upper right', title="Tissue", fontsize=14)
            legend.get_title().set_fontsize('16') #legend 'Title' fontsize

            fig.tight_layout()

            ax1.set_ylabel(r'Signal change [%]', fontsize=24)

            if stimDuration == 1:
                plt.title(f'{int(stimDuration)} second stimulation', fontsize=24,pad=10)
            else:
                plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24,pad=10)

            plt.savefig(f'./results/{sub}_stimDur-{int(stimDuration)}_{modality}_ERA-tissues.png', bbox_inches = "tight")

            plt.show()
