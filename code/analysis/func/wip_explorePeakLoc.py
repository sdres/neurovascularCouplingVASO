'''

Estimate layer profiles across geometric "columns" in the calcarine sulcus

'''

import nibabel as nb
import numpy as np
import os
import glob
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')

# Set subjects to work on
subs = ['sub-05','sub-06','sub-07','sub-09']
subs = ['sub-05']

subList = []
layerList = []
valList = []
voxelList = []
modalityList = []
stimDurList = []
columnList = []

for sub in subs:
    ROOT = f'/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/{sub}'
    SEGFOLDER = f'/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/{sub}/ses-01/anat/upsample'


    # Load perimeter data
    perimeterFile = f'{SEGFOLDER}/{sub}_rim-LH_perimeter_chunk.nii.gz'
    perimeterNii = nb.load(perimeterFile)
    perimeterData = perimeterNii.get_fdata()
    # Exclude rim of chunk
    perimeterData = (perimeterData == 1) * perimeterData

    # Load layer data
    layerFile = f'{SEGFOLDER}/{sub}_rim-LH_layers_equivol.nii.gz'
    layerNii = nb.load(layerFile)
    layerData = layerNii.get_fdata()
    layerData = layerData * perimeterData

    # Load column data
    columnFile = f'{SEGFOLDER}/{sub}_rim-LH_columns50.nii.gz'
    columnNii = nb.load(columnFile)
    columnData = columnNii.get_fdata()
    columnData = columnData * perimeterData

    for modality in ['vaso', 'bold']:

        for stimDur in [1,2,4,12,24]:
            # Load activation data
            activationFile = f'{ROOT}/statMaps/{sub}_{modality}_stim_{stimDur}s_registered_crop-toShpereLH.nii.gz'
            activationNii = nb.load(activationFile)
            activationData = activationNii.get_fdata()
            activationData = activationData * perimeterData

            columns = np.unique(columnData)[1:]
            layers = np.unique(layerData)[1:]

            # for column in columns:

            for layer in layers:

                idxColumn = columnData == column
                idxLayer = layerData == layer
                idxTotal = idxColumn * idxLayer

                val = np.mean(activationData[idxTotal])
                nrVox = np.sum(idxTotal)


                vals = activationData[idxLayer]
                val = np.mean(activationData[idxLayer])
                # val = np.mean(activationData[idxTotal])
                layerList.append(int(layer))
                valList.append(val)
                subList.append(sub)
                modalityList.append(modality)
                stimDurList.append(stimDur)
                columnList.append(int(column))

                    # for i, val in enumerate(vals):
                    #     layerList.append(int(layer))
                    #     valList.append(val)
                    #     voxelList.append(i)

# data = pd.DataFrame({'layer':layerList, 'val': valList, 'idx': voxelList})
data = pd.DataFrame({'sub':subList,'layer':layerList, 'val': valList, 'modality':modalityList, 'stimDur':stimDurList})
data = pd.DataFrame({'sub':subList,'layer':layerList, 'val': valList, 'modality':modalityList, 'stimDur':stimDurList, 'column':columnList})

for modality in ['bold', 'vaso']:
    tmp = data.loc[data['modality']== modality]
    # tmp = data.loc[(data['modality']== modality)&(data['column']== column)]
    sns.lineplot(data=tmp, x='layer', y = 'val', hue='stimDur')
    plt.show()


for column in columns:

    for modality in ['bold', 'vaso']:
        # tmp = data.loc[data['modality']== modality]
        tmp = data.loc[(data['modality']== modality)&(data['column']== column)&(data['sub']== 'sub-05')]
        sns.lineplot(data=tmp, x='layer', y = 'val', hue='stimDur')
        plt.title(f'{modality} {column}')
        plt.show()


# =============================================================================
# Extract data with metric depth
# =============================================================================


subList = []
metricList = []
valList = []
voxelList = []
modalityList = []




sns.scatterplot(data=data, x="layer", y="val")





columnList = []
layerList = []
valList = []
voxelList = []


for column in columns:
    for layer in layers:

        idxColumn = columnData == column
        idxLayer = layerData == layer
        idxTotal = idxColumn * idxLayer

        val = np.mean(activationData[idxTotal])
        nrVox = np.sum(idxTotal)

        columnList.append(int(column))
        layerList.append(int(layer))
        valList.append(val)
        voxelList.append(nrVox)


data = pd.DataFrame({'col': columnList, 'layer':layerList, 'val': valList, 'nrVox': voxelList})

sns.lineplot(data=data, x='layer', y = 'val')

for col in data['col'].unique():
    tmp = data.loc[data['col']==col]
    print(tmp['nrVox'])
    sns.lineplot(data=tmp, x='layer', y = 'val')
    plt.show()
