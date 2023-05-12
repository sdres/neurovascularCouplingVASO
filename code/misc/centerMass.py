"""Testing center of mass calculation for layer-activity"""

import numpy as np
import nibabel as nb
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']

subList = []
comList = []
durList = []
modalityList = []

for sub in subs:
    print('')
    print(sub)

    folder = f'/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/{sub}'
    metricFile = glob.glob(f'{folder}/ses-01/anat/upsample/{sub}_rim-LH_metric_equivol.nii*')[0]

    periFile = glob.glob(f'{folder}/ses-01/anat/upsample/{sub}_rim-LH_perimeter_chunk.nii*')[0]

    periData = nb.load(periFile).get_fdata()
    metricData = nb.load(metricFile).get_fdata()
    periData = (periData == 1).astype('int')
    metricData = metricData * periData

    for stimDur in [1, 2, 4, 12, 24]:

        for modality in ['bold', 'vaso']:
            actFile = f'{folder}/statMaps/glm_fsl/{sub}_{modality}_stim_{stimDur}s_registered_crop-toShpereLH.nii.gz'

            actData = nb.load(actFile).get_fdata()
            actData = actData * periData

            numerator = np.sum(np.multiply(actData, metricData))
            denominator = np.sum(actData)

            com = numerator / denominator

            print(f'The center of mass for stimDur: {stimDur}s is at depth {com}')

            subList.append(sub)
            comList.append(com)
            durList.append(stimDur)
            modalityList.append(modality)

data = pd.DataFrame({'subject': subList, 'cMass': comList, 'stimDur': durList, 'modality': modalityList})


for modality in ['bold', 'vaso']:

    plt.figure()
    tmp = data.loc[data['modality'] == modality]
    sns.catplot(
        data=tmp, x="stimDur", y="cMass", hue="subject",
        native_scale=True, zorder=1
    )

    sns.regplot(
        data=tmp, x="stimDur", y="cMass",
        scatter=False, truncate=False, order=2, color=".2",
    )
    plt.title(modality)
    plt.show()