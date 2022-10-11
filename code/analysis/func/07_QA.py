'''

Extracting QA metrics from data

'''

import nibabel as nb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

voxelList = []
modalityList = []
metricList = []
valList = []


for sub in ['sub-05']:
    for modality in ['vaso', 'bold']:
        for metric in ['tSNR', 'kurt', 'skew', 'mean']:
            nii = nb.load(f'{DATADIR}/{sub}/{sub}_task-stimulation_part-mag_{modality}_intemp_{metric}.nii.gz')
            data = nii.get_fdata()
            mask = nb.load(f'{DATADIR}/{sub}/v1Mask.nii.gz').get_fdata()

            tmp = data[mask.astype('bool')]

            for i, val in enumerate(tmp):
                voxelList.append(i)
                modalityList.append(modality)
                metricList.append(metric)
                valList.append(val[0])

data = pd.DataFrame({'voxel':voxelList, 'metric': metricList, 'modality': modalityList, 'data':valList})

plt.style.use('dark_background')


palette = {
    'bold': 'tab:orange',
    'vaso': 'tab:blue'}


for metric in ['tSNR']:


    tmp = data.loc[data['metric']==metric]

    fig, ax = plt.subplots()

    sns.kdeplot(data = tmp ,x = 'data', hue='modality',linewidth=2, palette=palette)

    plt.title(f'ROI {metric}',fontsize=24)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('voxel count',fontsize=24)
    plt.yticks([])

    if metric == 'tSNR':
        ticks = np.arange(5,31,5)
        plt.xticks(ticks, fontsize=14)

    plt.xlabel(f'{metric}',fontsize=20)

    #legend hack
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc='upper left', title='', fontsize=16)
    plt.savefig(f'results/{metric}.png',bbox_inches='tight')

    plt.show()
