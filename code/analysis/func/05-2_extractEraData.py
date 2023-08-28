"""Get event-related averages per stimulus duration"""

import os
import glob
import nibabel as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# Get TR
UPFACTOR = 4

# =============================================================================
# Extract voxel wise time-courses
# =============================================================================

STIMDURS = [1, 2, 4, 12, 24]

EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

MODALITIES = ['vaso', 'bold']
# MODALITIES = ['vaso']


SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
# SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-09']
# SUBS = ['sub-08']

timePointList = []
modalityList = []
valList = []
stimDurList = []
subList = []
depthList = []
dataTypeList = []

for sub in SUBS:
    print('')
    print(sub)

    subDir = f'{DATADIR}/{sub}'

    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    depthFile = glob.glob(f'{segFolder}/3layers_layers_equivol.nii*')[0]
    depthNii = nb.load(depthFile)
    depthData = depthNii.get_fdata()
    layers = np.unique(depthData)[1:]

    roisData = nb.load(glob.glob(f'{segFolder}/{sub}_rim-LH_perimeter_chunk.nii.gz')[0]).get_fdata()
    roiIdx = roisData == 1

    for stimDuration in [1, 2, 4, 12, 24]:
    # for stimDuration in [1]:
        print(f'stimDur: {stimDuration}s')
        for modality in ['vaso', 'bold']:
        # for modality in ['bold']:
            print(modality)
            frames = sorted(glob.glob(f'{DATADIR}/{sub}/ERAs/frames/{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}_'
                                      f'intemp_era-{stimDuration}s_sigChange-after_frame*_registered_crop.nii.gz'))

            print(f'Found {len(frames)} timepoints')

            # Zscore data
            nii = nb.load(frames[0])
            dataShape = nii.get_fdata().shape
            dataShape = np.append(dataShape, len(frames))
            data = np.zeros(dataShape)

            for n in range(len(frames)):
                nii = nb.load(frames[n])
                tmpData = nii.get_fdata()
                data[..., n] = tmpData

            mean = np.mean(data, axis=-1)
            stdDev = np.std(data, axis=-1)

            for j, frame in enumerate(frames):

                nii = nb.load(frame)
                data = nii.get_fdata()

                zscored = (data - mean)/stdDev

                for layer in layers:

                    layerIdx = depthData == layer
                    tmp = roiIdx*layerIdx

                    for dataType in ['raw', 'zscore']:
                        if dataType == 'raw':
                            val = np.mean(data[tmp])
                        if dataType == 'zscore':
                            val = np.mean(zscored[tmp])

                        valList.append(val)
                        subList.append(sub)
                        depthList.append(layer)
                        stimDurList.append(stimDuration)
                        modalityList.append(modality)
                        timePointList.append(j)
                        dataTypeList.append(dataType)

data = pd.DataFrame({'subject': subList,
                     'volume': timePointList,
                     'modality': modalityList,
                     'data': valList,
                     'layer': depthList,
                     'stimDur': stimDurList,
                     'dataType': dataTypeList})

data.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored.csv',
            sep=',',
            index=False)
#
# data.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_sub-08.csv',
#             sep=',',
#             index=False)
#
# data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored.csv', sep=',')
# data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_sub-08.csv', sep=',')

# Equalize mean between first and second set
tr = 3.1262367768477795/4
EVENTDURS = {'shortITI': (np.array([11, 14, 18, 32, 48])/tr).astype('int'),
             'longITI': (np.array([21, 24, 28, 42, 64])/tr).astype('int')}

STIMDURS = [1, 2, 4, 12, 24]

equalized = pd.DataFrame()

for sub in data['subject'].unique():
    for dataType in ['raw', 'zscore']:
        for modality in ['vaso', 'bold']:
        # for modality in ['vaso']:
                for layer in data['layer'].unique():
                for i, stimDur in enumerate(STIMDURS):
                    tmp = data.loc[(data['dataType'] == dataType)
                                   & (data['subject'] == sub)
                                   & (data['modality'] == modality)
                                   & (data['layer'] == layer)
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

equalized.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalized.csv',
            sep=',',
            index=False)


equalized = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalized.csv', sep=',')

normalized = pd.DataFrame()

for sub in SUBS:
    for modality in ['bold', 'vaso']:
    # for modality in ['vaso']:
        for stimDuration in [1., 2., 4., 12., 24.]:
            for layer in [1, 2, 3]:
                for dataType in ['raw', 'zscore']:

                    tmp = equalized.loc[(equalized['stimDur'] == stimDuration)
                                        & (equalized['layer'] == layer)
                                        & (equalized['modality'] == modality)
                                        & (equalized['dataType'] == dataType)
                                        & (equalized['subject'] == sub)]


                    if dataType == 'raw':
                        # Get value of first volume for given layer
                        val = np.mean(tmp.loc[(tmp['volume'] == 0)]['data'])
                        # Normalize to that value
                        tmp['data'] = tmp['data'] - val

                    normalized = pd.concat((normalized, tmp))

normalized.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalizedNormalized.csv',
            sep=',',
            index=False)

# normalized.to_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_rawPlusZscored_equalizedNormalized_sub-08.csv',
#             sep=',',
#             index=False)