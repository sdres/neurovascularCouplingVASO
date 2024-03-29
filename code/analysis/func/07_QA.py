"""

Extracting QA metrics from data

"""

import nibabel as nb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

voxelList = []
modalityList = []
metricList = []
valList = []


for sub in ['sub-06']:
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

data = pd.DataFrame({'voxel': voxelList, 'metric': metricList, 'modality': modalityList, 'data': valList})

plt.style.use('dark_background')


palette = {
    'bold': 'tab:orange',
    'vaso': 'tab:blue'}

for sub in ['sub-06']:
    for metric in ['tSNR']:


        tmp = data.loc[data['metric']==metric]

        fig, ax = plt.subplots()

        sns.kdeplot(data = tmp ,x = 'data', hue='modality',linewidth=2, palette=palette)

        plt.title(f'ROI {metric}', fontsize=24)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('voxel count', fontsize=24)
        plt.yticks([])

        if metric == 'tSNR':
            ticks = np.arange(5, 31, 5)
            plt.xticks(ticks, fontsize=14)

        plt.xlabel(f'{metric}', fontsize=20)

        #legend hack
        old_legend = ax.legend_
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()
        ax.legend(handles, labels, loc='upper left', title='', fontsize=16)
        plt.savefig(f'results/{sub}_{metric}.png',bbox_inches='tight')

        plt.show()

# ============================================================================================================
# Highres ROIs
# ============================================================================================================

MODALITIES = ['bold', 'vaso']
subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']

subList = []
valList = []
sesList = []
modalityList = []
voxList = []
metricList = []

for sub in subs:
    print(f'Processing {sub}')
    subFolder = f'{DATADIR}/{sub}'
    statFolder = f'{DATADIR}/{sub}/statMaps/glm_fsl'
    segFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'

    roisData = nb.load(glob.glob(f'{segFolder}/{sub}_rim-LH_perimeter_chunk.nii.gz')[0]).get_fdata()
    roiIdx = roisData == 1

    for modality in MODALITIES:
        for metric in ['tSNR']:
        # for metric in ['tSNR', 'mean', 'skew', 'kurt']:
                files = sorted(glob.glob(f'{subFolder}/ses-*/func/{sub}_ses-*_task-stimulation_run-avg_part-mag_'
                                         f'{modality}_intemp_{metric}_registered_crop-toShpereLH.nii.gz')
                               )

            for file in files:
                for j in range(1, 6):
                    if f'ses-0{j}' in file:
                        ses = f'ses-0{j}'

                data = nb.load(file).get_fdata()

                dataMasked = data[roiIdx]

                for i, val in enumerate(dataMasked):

                    subList.append(sub)
                    valList.append(val)
                    sesList.append(ses)
                    modalityList.append(modality)
                    voxList.append(i)
                    metricList.append(metric)

# Save data to dataframe
data = pd.DataFrame({'subject': subList,
                     'session': sesList,
                     'value': valList,
                     'voxelID': voxList,
                     'modality': modalityList,
                     'metric': metricList}
                    )
data.to_csv(f'results/QA_metrics.csv', index=False, sep=',')


plt.style.use('dark_background')


palette = {
    'bold': 'tab:orange',
    'vaso': 'tab:blue'}

for metric in data['metric'].unique():
    fig, ax = plt.subplots()

    tmp = data.loc[data['metric'] == metric]

    sns.kdeplot(data=tmp, x='value', hue='modality', linewidth=2, palette=palette)

    # plt.title(f'ROI {metric}', fontsize=24)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Voxel count', fontsize=24)
    plt.yticks([])

    ticks = np.arange(0, 61, 10)
    plt.xticks(ticks, fontsize=18)
    plt.xlim([0, 60])

    plt.xlabel(f'{metric}', fontsize=24)

    #legend hack
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    # labels = [t.get_text() for t in old_legend.get_texts()]
    labels = ['BOLD', 'VASO']
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc='upper right', title='', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/QA/group_{metric}.png',
        bbox_inches="tight")
    plt.show()


# Plot single subs
for metric in data['metric'].unique():
    for modality in MODALITIES:
        fig, ax = plt.subplots()
        tmp = data.loc[(data['metric'] == metric) & (data['modality'] == modality)]
        sns.kdeplot(data=tmp, x='value', hue='subject', linewidth=2)

        plt.title(f'{modality} {metric}', fontsize=24)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('voxel count', fontsize=24)
        plt.yticks([])

        # ticks = np.arange(0, 61, 10)
        # plt.xticks(ticks, fontsize=14)
        # plt.xlim([0, 60])

        plt.xlabel(f'{metric}', fontsize=20)

        #legend hack
        old_legend = ax.legend_
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()
        ax.legend(handles, labels, loc='upper right', title='', fontsize=16)
        plt.tight_layout()
        # plt.savefig(f'/Users/sebastiandresbach/Desktop/tSNR.png', bbox_inches='tight')

        plt.show()


# Plot single sessions
for sub in data['subject'].unique():
    for metric in data['metric'].unique():
        for modality in MODALITIES:
            fig, ax = plt.subplots()
            tmp = data.loc[(data['metric'] == metric) & (data['modality'] == modality) & (data['subject'] == sub)]
            sns.kdeplot(data=tmp, x='value', hue='session', linewidth=2)

            plt.title(f'{sub} {modality} {metric}', fontsize=24)
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylabel('voxel count', fontsize=24)
            plt.yticks([])

            if modality == 'vaso':
                ticks = np.arange(0, 31, 5)
                plt.xticks(ticks, fontsize=14)
                plt.xlim([0, 30])

            if modality == 'bold':
                ticks = np.arange(0, 61, 10)
                plt.xticks(ticks, fontsize=14)
                plt.xlim([0, 60])

            plt.xlabel(f'{metric}', fontsize=20)

            # Calculate and plot mean
            m = np.mean(tmp['value'])
            ax.axvline(m, linestyle='--', color='white', label='mean')

            # Get ylim
            lims = ax.get_ylim()
            # get lowest 5% point
            p = (lims[1] / 100) * 95

            # ax.text(m+0.3, p, r'mean $\approx$ {}'.format(round(m)), color='white')
            ax.text(m+0.3, p, f'mean={m:.1f}', color='white')

            #legend hack
            old_legend = ax.legend_
            handles = old_legend.legend_handles
            labels = [t.get_text() for t in old_legend.get_texts()]
            title = old_legend.get_title().get_text()
            ax.legend(handles, labels, loc='upper right', title='', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'/Users/sebastiandresbach/github/neurovascularCouplingVASO/results/QA/{sub}_{modality}_{metric}.png', bbox_inches='tight')

            plt.show()

# ==================================================================================================================
# Make mean image of session 1 nulled and not nulled

boldFile = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/func/sub-06_ses-01_task-stimulation_run-avg_part-mag_bold.nii'
nulledFile = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/func/sub-06_ses-01_task-stimulation_run-avg_part-mag_cbv.nii'

nulledNii = nb.load(nulledFile)
nulledData = nulledNii.get_fdata()

nulledMean = np.mean(nulledData, axis=-1)
img = nb.Nifti1Image(nulledMean, affine=nulledNii.affine, header=nulledNii.header)
nb.save(img, '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/func/sub-06_ses-01_task-stimulation_run-avg_part-mag_cbv_mean.nii')

nulledNii = nb.load(boldFile)
nulledData = nulledNii.get_fdata()

nulledMean = np.mean(nulledData, axis=-1)
img = nb.Nifti1Image(nulledMean, affine=nulledNii.affine, header=nulledNii.header)
nb.save(img, '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/func/sub-06_ses-01_task-stimulation_run-avg_part-mag_bold_mean.nii')

