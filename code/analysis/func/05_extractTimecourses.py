''' Get event-related avergaes per stimulus duration '''

import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy import interpolate

import sys
sys.path.append('./code/misc')
from findTr import *


ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'
# Define subjects to work on
subs = ['sub-08']
# sessions = ['ses-01', 'ses-03']
sessions = ['ses-01']


# Get TR
UPFACTOR = 4


# =============================================================================
# Extract mean timecourses
# =============================================================================

for sub in subs:
    for ses in sessions:

        for modality in ['vaso', 'bold']:
        # for modality in ['bold']:

            run = f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz'
            # run = f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}.nii'

            nii = nb.load(run)
            header = nii.header
            data = nii.get_fdata()

            mask = nb.load(f'{DATADIR}/{sub}/v1Mask.nii.gz').get_fdata()

            mask_mean = np.mean(data[:, :, :][mask.astype(bool)], axis=0)

            np.save(f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp_timecourse',
                    mask_mean
                    )

# =============================================================================
# Extract voxel wise timecourses
# =============================================================================

SUBS = ['sub-05']
STIMDURS = [1, 2, 4, 12, 24]
# STIMDURS = [24]
EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

# EVENTDURS = np.array([48])
# EVENTDURS = np.array([64])

MODALITIES = ['vaso', 'bold']
# MODALITIES = ['bold']

for sub in SUBS:
    eraDir = f'{DATADIR}/{sub}/ERAs'  # Location of functional data
    # Make output folder if it does not exist already
    if not os.path.exists(eraDir):
        os.makedirs(eraDir)

    # =========================================================================
    # Look for sessions
    # Collectall runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))

    # Get dimensions
    nii = nb.load(allRuns[0])
    dims = nii.header['dim'][1:4]

    tr = findTR(f'code/stimulation/{sub}/ses-01/{sub}_ses-01_run-01_neurovascularCoupling.log')
    # print(f'Effective TR: {tr} seconds')
    tr = tr/UPFACTOR
    # print(f'Nominal TR will be: {tr} seconds')

    # Initialte list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1,6):  # We had a maximum of 5 sessions
            if f'ses-0{i}' in run:
                sessions.append(f'ses-0{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')
    # sessions = ['ses-03']

    for modality in MODALITIES:
        print(f'\nProcessing {modality}')

        for j, stimDur in enumerate(STIMDURS):
            print(f'Processing stim duration: {stimDur}s')

            dimsTmp = np.append(dims, int(EVENTDURS['longITI'][j]/tr))

            tmp = np.zeros(dimsTmp)
            divisor = np.zeros(dimsTmp)

            for ses in sessions:
                print(f'processing {ses}')

                # Set run name
                run = f'{DATADIR}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz'

                # Set design file of first run in session
                design = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-01_part-mag_bold_events.tsv')
                # Load run
                nii = nb.load(run)
                header = nii.header
                affine = nii.affine
                # Load data as array
                data = nii.get_fdata()

                if data.shape[-1] > 240:
                    iti = 'longITI'
                else:
                    iti = 'shortITI'

                mean = np.mean(data, axis = -1)

                tmpMean = np.zeros(data.shape)

                for i in range(tmpMean.shape[-1]):
                    tmpMean[...,i] = mean

                data = (np.divide(data, tmpMean) - 1) * 100

                onsets = design.loc[design['trial_type'] == f'stim {stimDur}s']

                runDims = np.append(dims, int(EVENTDURS[iti][j]/tr))
                tmpRun = np.zeros(runDims)

                for onset in onsets['onset']:

                    startTR = int(onset/tr)
                    endTR = startTR + int(EVENTDURS[iti][j]/tr)
                    nrTimepoints = endTR - startTR

                    tmpRun += data[..., startTR: endTR]

                tmpRun /= len(onsets['onset'])

                tmp = np.add(tmp[..., :nrTimepoints], tmpRun)
                divisor[:,:,:,:nrTimepoints] += 1


            tmp = np.divide(tmp, divisor)
            if modality == 'vaso':
                tmp = tmp * -1
            img = nb.Nifti1Image(tmp, header = header, affine = affine)
            nb.save(img, f'{DATADIR}/{sub}/ERAs/{sub}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDur}s_sigChange.nii.gz')

files = glob.glob(f'{eraDir}/*sigChange.nii.gz')
for file in files:
    base = file.split('.')[0]
    command = f'fslmaths {file} -mul {DATADIR}/{sub}/{sub}_brainMask.nii.gz {base}_masked.nii.gz'
    subprocess.run(command,shell=True)

# Test whether extracted ERAS give same results

SUBS = ['sub-05','sub-06']
SUBS = ['sub-06']

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []
depthList = []

for sub in SUBS:
    subDir = f'{DATADIR}/{sub}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    depthFile = f'{segFolder}/3layers_layers_equivol.nii'
    depthNii = nb.load(depthFile)
    depthData = depthNii.get_fdata()
    layers = np.unique(depthData)[1:]

    # roisData = nb.load(f'{roiFolder}/sub-05_vaso_stimulation_registered_crop_largestCluster_bin_UVD_max_filter.nii.gz').get_fdata()
    roisData = nb.load(f'{segFolder}/{sub}_rim_perimeter_chunk.nii.gz').get_fdata()
    roiIdx = roisData == 1


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
            #
            # # Because we want the % signal-change, we need the mean
            # # of the voxels we are looking at.
            # # mask_mean = np.mean(mriData)
            #
            # # # Or we can normalize to the rest priods only
            # # restTRs = np.ones(mriData.shape, dtype=bool)
            # #
            # # for startTR, stopTR in zip(allStartTRs,allStopTRs):
            # #     restTRs[startTR:stopTR] = False
            # #
            # # mask_mean = np.mean(mriData[restTRs])
            #
            # # for i, (start, end) in enumerate(zip(startTRs, stopTRs)):
            # #     tmp = ((( mriData[int(start):int(end)] / mask_mean) - 1) * 100)
            # #     # tmp = ((( mriData[int(start):int(end)] / mask_mean)) * 100)
            # #
            # #     trials[i,:] = tmp
            # if modality == 'vaso':
            #     mask_mean = -mask_mean
            #
            # for j, item in enumerate(mask_mean):
            #     timePointList.append(j)
            #     modalityList.append(modality)
            #     valList.append(item)
            #     stimDurList.append(stimDuration)
            #     subList.append(sub)


data = pd.DataFrame({'subject': subList, 'volume': timePointList, 'modality': modalityList, 'data': valList, 'layer':depthList, 'stimDur':stimDurList})
# data = pd.DataFrame({'subject': subList, 'volume': timePointList, 'modality': modalityList, 'data': valList, 'stimDur': stimDurList})

data.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/{sub}_task-stimulation_responsestest.csv', sep = ',', index=False)



palettesLayers = {'vaso':['#55a8e2','#aad4f0','#ffffff','#FF0000'],
'bold':['#ff8c26', '#ffd4af','#ffffff','#FF0000']}
layerNames = ['deep', 'middle', 'superficial','vein']

# for interpolationType in ['linear', 'cubic']:
for interpolationType in ['linear']:
    # data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/{sub}_task-stimulation_responses.csv', sep = ',')
    # data = data.loc[data['interpolation']==interpolationType]
    for modality in ['bold', 'vaso']:
    # for modality in ['vaso']:

        for stimDuration in [1., 2., 4., 12., 24.]:
            fig, (ax1) = plt.subplots(1,1,figsize=(7.5,5))

            # for modality in ['bold', 'vaso']:

            for layer in [1,2,3,4]:

                val = np.mean(data.loc[(data['stimDur'] == stimDuration)
                                     & (data['volume'] == 0)
                                     & (data['layer'] == layer)]['data']
                                     )

                tmp = data.loc[(data['stimDur'] == stimDuration)&(data['layer'] == layer)&(data['modality'] == modality)&(data['subject'] == 'sub-06')]

                # if val > 0:
                #     tmp['data'] = tmp['data'] - val
                # if val < 0:
                    # tmp['data'] = tmp['data'] + val
                # tmp['data'] = tmp['data'] - val
                nrVols = len(np.unique(tmp['volume']))

                # ax1.set_xticks(np.arange(-1.5,3.6))
                if modality == 'vaso':
                    ax1.set_ylim(-2.1,4.1)
                if modality == 'bold':
                    ax1.set_ylim(-4.1,6.1)
                sns.lineplot(ax=ax1,
                             data = tmp,
                             x = "volume",
                             y = "data",
                             color = palettesLayers[modality][layer-1],
                             linewidth = 3,
                             # ci=None,
                             label = layerNames[layer-1],
                             )

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

            legend = ax1.legend(loc='upper right', title="Layer", fontsize=14)
            legend.get_title().set_fontsize('16') #legend 'Title' fontsize

            fig.tight_layout()

            ax1.set_ylabel(r'Signal change [%]', fontsize=24)

            if stimDuration == 1:
                plt.title(f'{int(stimDuration)} second stimulation', fontsize=24,pad=10)
            else:
                plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24,pad=10)

            plt.savefig(f'./results/{sub}_stimDur-{int(stimDuration)}_{modality}_ERA-layers.png', bbox_inches = "tight")

            plt.show()




# =============================================================================
# extract from vessels
# =============================================================================

SUBS = ['sub-06']

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []
vesselList = []

# vessels = {1:'veins', 2:'arteries'}

for sub in SUBS:
    subDir = f'{DATADIR}/{sub}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    vesselFile = f'/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/finalVeins_registered_crop.nii.gz'
    vesselNii = nb.load(vesselFile)
    vesselData = vesselNii.get_fdata()
    vesseTypes = np.unique(vesselData)[1:]
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

                for vesselType in vesseTypes:

                    idx = vesselData == vesselType
                    # tmp = roiIdx*layerIdx

                    val = np.mean(data[idx])

                    if modality == 'bold':
                        valList.append(val)
                    if modality == 'vaso':
                        valList.append(val)

                    subList.append(sub)
                    depthList.append(4)
                    stimDurList.append(stimDuration)
                    modalityList.append(modality)
                    timePointList.append(j)


data = pd.DataFrame({'subject': subList, 'volume': timePointList, 'modality': modalityList, 'data': valList, 'vessel':vesselList, 'stimDur':stimDurList})
# data = pd.DataFrame({'subject': subList, 'volume': timePointList, 'modality': modalityList, 'data': valList, 'stimDur': stimDurList})

data.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/{sub}_task-stimulation_responsesAcrossVessels.csv', sep = ',', index=False)



tmp = data.loc[(data['stimDur'] == 24)&(data['modality'] == 'vaso')&(data['subject'] == 'sub-06')]

sns.lineplot(
             data = tmp,
             x = "volume",
             y = "data",
             hue='vessel',
             linewidth = 3
             # ci=None,
             )

palettesLayers = {'vaso':['#55a8e2','#FF0000'],
'bold':['#ff8c26','#FF0000']}
layerNames = ['GM','vein']

# for interpolationType in ['linear', 'cubic']:
for interpolationType in ['linear']:
    # data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/{sub}_task-stimulation_responses.csv', sep = ',')
    # data = data.loc[data['interpolation']==interpolationType]
    for modality in ['bold', 'vaso']:
    # for modality in ['vaso']:

        for stimDuration in [1., 2., 4., 12., 24.]:
            fig, (ax1) = plt.subplots(1,1,figsize=(7.5,5))

            # for modality in ['bold', 'vaso']:

            for layer in [1,2]:

                val = np.mean(data.loc[(data['stimDur'] == stimDuration)
                                     & (data['volume'] == 0)
                                     & (data['layer'] == layer)]['data']
                                     )

                if layer == 1:
                    tmp = data.loc[(data['stimDur'] == stimDuration)&(data['layer'] != 4)&(data['modality'] == modality)&(data['subject'] == 'sub-06')]

                if layer == 2:
                    tmp = data.loc[(data['stimDur'] == stimDuration)&(data['layer'] == 4)&(data['modality'] == modality)&(data['subject'] == 'sub-06')]

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
                             color = palettesLayers[modality][layer-1],
                             linewidth = 3,
                             # ci=None,
                             label = layerNames[layer-1],
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

# import os
# import glob
# import nibabel as nb
# import numpy as np
# import subprocess
# import pandas as pd
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# plt.style.use('dark_background')
#
# palette = {
#     'bold': 'tab:orange',
#     'vaso': 'tab:blue'}
#
#
# data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/{sub}_task-stimulation_responsestest.csv', sep = ',')
#
# # for stimDuration in [1, 2, 4, 12, 24]:
# for stimDuration in [12]:
#
#     fig, (ax1) = plt.subplots(1,1,figsize=(7.5,5))
#
#     for modality in ['bold', 'vaso']:
#
#         val = np.mean(data.loc[(data['stimDur'] == stimDuration)
#                              & (data['volume'] == 0)
#                              & (data['modality'] == modality)]['data']
#                              )
#
#         tmp = data.loc[(data['stimDur'] == stimDuration)&(data['modality'] == modality)]
#
#         # if val > 0:
#         #     tmp['data'] = tmp['data'] - val
#         # if val < 0:
#             # tmp['data'] = tmp['data'] + val
#
#         nrVols = len(np.unique(tmp['volume']))
#
#         sns.lineplot(ax=ax1,
#                      data = tmp,
#                      x = "volume",
#                      y = "data",
#                      color = palette[modality],
#                      linewidth = 3,
#                      # ci=None,
#                      label = modality,
#                      )
#
#     # Prepare x-ticks
#     ticks = np.linspace(0, nrVols, 10)
#     labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)
#
#     ax1.set_yticks(np.arange(-0.25, 3.51, 0.5))
#
#     ax1.yaxis.set_tick_params(labelsize=18)
#     ax1.xaxis.set_tick_params(labelsize=18)
#
#     # tweak x-axis
#     ax1.set_xticks(ticks[::2])
#     ax1.set_xticklabels(labels[::2],fontsize=18)
#     ax1.set_xlabel('Time [s]', fontsize=24)
#
#     # draw lines
#     ax1.axvspan(0, stimDuration / 0.7808410714285715, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation')
#     # get value of first timepoint
#
#     ax1.axhline(0,linestyle = '--', color = 'white')
#
#     legend = ax1.legend(loc='upper right', title="Modalities", fontsize=14)
#     legend.get_title().set_fontsize('16') #legend 'Title' fontsize
#
#     fig.tight_layout()
#
#     ax1.set_ylabel(r'Signal change [%]', fontsize=24)
#
#     if stimDuration == 1:
#         plt.title(f'{int(stimDuration)} second stimulation', fontsize=24,pad=10)
#     else:
#         plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24,pad=10)
#
#     # plt.savefig(f'./results/{sub}_stimDur-{int(stimDuration)}_intemp-{interpolationType}.png', bbox_inches = "tight")
#
#     plt.show()
