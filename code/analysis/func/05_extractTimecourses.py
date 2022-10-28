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
EVENTDURS = np.array([11, 14, 20, 32, 48])
# EVENTDURS = np.array([48])
# EVENTDURS = np.array([64])

MODALITIES = ['vaso', 'bold']
# MODALITIES = ['bold']

for sub in SUBS:
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

        for stimDur, eventDur in zip(STIMDURS, EVENTDURS):
            print(f'Processing stim duration: {stimDur}s')

            dimsTmp = np.append(dims, int(eventDur/tr))

            tmp = np.zeros(dimsTmp)

            for ses in sessions:
                print(f'processing {ses}')
                tmpRun = np.zeros(dimsTmp)

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

                mean = np.mean(data, axis = -1)

                tmpMean = np.zeros(data.shape)

                for i in range(tmpMean.shape[-1]):
                    tmpMean[...,i] = mean

                data = (np.divide(data, tmpMean) - 1) * 100

                onsets = design.loc[design['trial_type'] == f'stim {stimDur}s']

                for onset in onsets['onset']:
                    startTR = int(onset/tr)
                    endTR = startTR + int(eventDur/tr)
                    # test = data[..., startTR:endTR]

                    tmpRun += data[..., startTR: endTR]

                tmpRun /= len(onsets['onset'])

                # tmpMean = np.zeros(tmpRun.shape)
                #
                # for i in range(tmpRun.shape[-1]):
                #     tmpMean[...,i] = mean
                #
                # tmpRun = ((np.divide(tmp, tmpMean) - 1) * 100)

                tmp += tmpRun

            tmp /= len(sessions)

            img = nb.Nifti1Image(tmp, header = header, affine = affine)
            nb.save(img, f'{DATADIR}/{sub}/{sub}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDur}s_sigChange-after.nii.gz')


# Test whether extracted ERAS give same results

SUBS = ['sub-05','sub-06']
SUBS = ['sub-06']

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []

for sub in SUBS:
    subDir = f'{DATADIR}/{sub}'


    for stimDuration in [24]:

        for modality in ['vaso', 'bold']:
        # for modality in ['bold']:
            file = f'{DATADIR}/{sub}/{sub}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-{stimDuration}s_sigChange-after.nii.gz'

            nii = nb.load(file)

            header = nii.header

            data = nii.get_fdata()
            # print(data.shape)

            mask = nb.load(f'{DATADIR}/{sub}/v1Mask.nii.gz').get_fdata()

            mask_mean = np.mean(data[:, :, :][mask.astype(bool)], axis=0)

            # Because we want the % signal-change, we need the mean
            # of the voxels we are looking at.
            # mask_mean = np.mean(mriData)

            # # Or we can normalize to the rest priods only
            # restTRs = np.ones(mriData.shape, dtype=bool)
            #
            # for startTR, stopTR in zip(allStartTRs,allStopTRs):
            #     restTRs[startTR:stopTR] = False
            #
            # mask_mean = np.mean(mriData[restTRs])

            # for i, (start, end) in enumerate(zip(startTRs, stopTRs)):
            #     tmp = ((( mriData[int(start):int(end)] / mask_mean) - 1) * 100)
            #     # tmp = ((( mriData[int(start):int(end)] / mask_mean)) * 100)
            #
            #     trials[i,:] = tmp
            if modality == 'vaso':
                mask_mean = -mask_mean

            for j, item in enumerate(mask_mean):
                timePointList.append(j)
                modalityList.append(modality)
                valList.append(item)
                stimDurList.append(stimDuration)
                subList.append(sub)


data = pd.DataFrame({'subject': subList, 'volume': timePointList, 'modality': modalityList, 'data': valList, 'stimDur': stimDurList})

data.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/{sub}_task-stimulation_responsestest.csv', sep = ',', index=False)




import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('dark_background')

palette = {
    'bold': 'tab:orange',
    'vaso': 'tab:blue'}


data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/{sub}_task-stimulation_responsestest.csv', sep = ',')

for stimDuration in [1, 2, 4, 12, 24]:

    fig, (ax1) = plt.subplots(1,1,figsize=(7.5,5))

    for modality in ['bold', 'vaso']:

        val = np.mean(data.loc[(data['stimDur'] == stimDuration)
                             & (data['volume'] == 0)
                             & (data['modality'] == modality)]['data']
                             )

        tmp = data.loc[(data['stimDur'] == stimDuration)&(data['modality'] == modality)]

        # if val > 0:
        #     tmp['data'] = tmp['data'] - val
        # if val < 0:
            # tmp['data'] = tmp['data'] + val

        nrVols = len(np.unique(tmp['volume']))

        sns.lineplot(ax=ax1,
                     data = tmp,
                     x = "volume",
                     y = "data",
                     color = palette[modality],
                     linewidth = 3,
                     # ci=None,
                     label = modality,
                     )

    # Prepare x-ticks
    ticks = np.linspace(0, nrVols, 10)
    labels = (np.linspace(0, nrVols, 10)*0.7808410714285715).round(decimals=1)

    ax1.set_yticks(np.arange(-0.25, 3.51, 0.5))

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

    legend = ax1.legend(loc='upper right', title="Modalities", fontsize=14)
    legend.get_title().set_fontsize('16') #legend 'Title' fontsize

    fig.tight_layout()

    ax1.set_ylabel(r'Signal change [%]', fontsize=24)

    if stimDuration == 1:
        plt.title(f'{int(stimDuration)} second stimulation', fontsize=24,pad=10)
    else:
        plt.title(f'{int(stimDuration)} seconds stimulation', fontsize=24,pad=10)

    # plt.savefig(f'./results/{sub}_stimDur-{int(stimDuration)}_intemp-{interpolationType}.png', bbox_inches = "tight")

    plt.show()
