"""Find peak layer for each stim duration"""

import numpy as np
import nibabel as nb
import glob
import pandas as pd
import seaborn as sns

subs = ['sub-05', 'sub-06', 'sub-07', 'sub-09']

subList = []
peakList = []
durList = []

for sub in subs:
    print('')
    print(sub)

    folder = f'/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/{sub}'
    layerFile = glob.glob(f'{folder}/ses-01/anat/upsample/{sub}_rim-LH_layers_equivol.nii*')[0]

    periFile = glob.glob(f'{folder}/ses-01/anat/upsample/{sub}_rim-LH_perimeter_chunk.nii*')[0]

    periData = nb.load(periFile).get_fdata()
    layerData = nb.load(layerFile).get_fdata()
    periData = (periData == 1).astype('int')
    layerData = layerData * periData

    layers = np.unique(layerData)[1:]

    for stimDur in [1, 2, 4, 12, 24]:
        actFile = f'{folder}/statMaps/glm_fsl/{sub}_vaso_stim_{stimDur}s_registered_crop-toShpereLH.nii.gz'

        actData = nb.load(actFile).get_fdata()

        peakLayer = 0
        maxLayerVal = 0

        for layer in layers:
            roi = (layerData == layer).astype('int')
            layerAct = actData * roi

            layerMean = np.mean(layerAct)
            if layerMean > maxLayerVal:
                maxLayerVal = layerMean
                peakLayer = layer


        subList.append(sub)
        peakList.append(peakLayer)
        durList.append(stimDur)

data = pd.DataFrame({'subject': subList, 'peak': peakList, 'stimDur': durList})

sns.catplot(
    data=data, x="stimDur", y="peak", hue="subject",
    native_scale=True, zorder=1
)

sns.regplot(
    data=data, x="stimDur", y="peak",
    scatter=False, truncate=False, order=2, color=".2",
)


# =============================================================================
# For ERAs
# =============================================================================
# Load data
data = pd.read_csv(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/ERAs_zscored.csv', sep=',')
subs = ['sub-05']
layerNames = ['deep', 'middle', 'superficial', 'vein']

layerList = []
peakTPList = []
peakValList = []
modalityList = []
stimDurList = []

# for sub in subs:
for modality in ['vaso']:
    for stimDur in data['stimDur'].unique():
        # Get max timepoints for stim duration
        maxTP = np.max(data.loc[(data['stimDur'] == stimDur)]['volume'].to_numpy())
        for layer in data['layer'].unique():
            # Find peak time-point
            maxVal = 0
            peakTP = 0
            for timepoint in range(maxTP):
                tmp = data.loc[(data['stimDur'] == stimDur)
                               & (data['layer'] == layer)
                               & (data['modality'] == modality)
                               # & (data['subject'] == sub)
                               & (data['volume'] == timepoint)]

                val = tmp.iloc[0]['data']
                if val >= maxVal:
                    maxVal = val
                    peakTP = timepoint

            layerList.append(layer)
            peakTPList.append(peakTP)
            peakValList.append(maxVal)
            modalityList.append(modality)
            stimDurList.append(stimDur)

peakData = pd.DataFrame({'layer': layerList,
                         'peakTP': peakTPList,
                         'peakVal': peakValList,
                         'modality': modalityList,
                         'stimDur': stimDurList})
