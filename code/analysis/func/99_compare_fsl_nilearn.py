"""

Compare the statistical values computed with nilearn versus fsl

"""


import pandas as pd
import numpy as np
import nibabel as nb
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Set data path
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# =======================================================
# Collect data

# Set subjects to work on
subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
# subs = ['sub-06','sub-07']
# subs = ['sub-09']

MODALITIES = ['bold', 'vaso']
analysisPrograms = ['fsl', 'nilearn']

subList = []
depthList = []
valList = []
stimList = []
modalityList = []
programList = []

for sub in subs:
    print(f'Processing {sub}')
    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    depthFile = glob.glob(f'{segFolder}/{sub}_rim-LH*layers_equivol.nii*')[0]
    depthNii = nb.load(depthFile)
    depthData = depthNii.get_fdata()
    layers = np.unique(depthData)[1:]

    roisData = nb.load(glob.glob(f'{segFolder}/{sub}_rim-LH_perimeter_chunk.nii*')[0]).get_fdata()

    roiIdx = roisData == 1

    for program in analysisPrograms:
        statFolder = f'{DATADIR}/{sub}/statMaps/glm_{program}'

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
                    programList.append(program)

data = pd.DataFrame({'subject': subList, 'depth': depthList, 'value': valList, 'stim': stimList, 'modality': modalityList, 'program': programList})


# =======================================================
# Plotting
# =======================================================

plt.style.use('dark_background')


for modality in ['vaso', 'bold']:
    for stimDur in [1, 2, 4, 12, 24]:
    # for stimDur in [24]:

        fig, ax = plt.subplots()
        tmp = data.loc[(data['modality'] == modality) & (data['stim'] == stimDur) & (data['subject'] != 'sub-08')]

        sns.lineplot(data=tmp, x='depth', y='value', hue='program', linewidth=2)

        lim = ax.get_ylim()
        lim[0].round()

        ax.set_yticks(np.linspace(lim[0].round(), lim[1].round(), 5))

        plt.ylabel(f'z-score', fontsize=20)

        plt.title(f"{modality} {stimDur}s", fontsize=24, pad=20)
        plt.xlabel('WM                                              CSF', fontsize=20)
        plt.xticks([])
        yLimits = ax.get_ylim()

        plt.yticks(fontsize=18)

        plt.legend(loc='upper left')
        legend = plt.legend(fontsize=14, loc='lower center')

        title = legend.get_title()
        title.set_fontsize(14)
        plt.tight_layout()
        plt.savefig(f'/Users/sebastiandresbach/Desktop/sub-all_{modality}_stimDur-{stimDur}s_zScoreProfile_fslVsNilearn.png',
                    bbox_inches="tight")
        plt.show()

