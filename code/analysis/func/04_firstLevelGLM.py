'''

Running first level GLM in FSL using Nilearn

'''

import nibabel as nb
import nilearn
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix
import glob
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
import os
import re
import sys
import subprocess
# Define current dir
ROOT = os.getcwd()
sys.path.append('./code/misc')

from findTr import *

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

subs = ['sub-05', 'sub-06']
subs = ['sub-07']

drift_model = 'Cosine'  # We use a discrete cosine transform to model signal drifts.
high_pass = .01  # The cutoff for the drift model is 0.01 Hz.
hrf_model = 'spm'  # The hemodynamic response function is the SPM canonical one.


for sub in subs:

    funcDir = f'{ROOT}/derivatives/{sub}'
    # make folder to dump statistocal maps
    statFolder = f'{funcDir}/statMaps'

    if not os.path.exists(statFolder):
        os.makedirs(statFolder)
        print("Statmap directory is created")

    # Check for sessions with long ITI
    longITIses = 0
    allRuns = sorted(glob.glob(f'{funcDir}/ses-*/func/{sub}_ses-*_task-stimulation_run-01_part-mag_cbv.nii.gz'))

    for run in allRuns:
        base = os.path.basename(run).rsplit('.', 2)[0]
        nii = nb.load(run)
        nrTRs = nii.header['dim'][4]

        if nrTRs > 240:
            longITIses = longITIses+1

            for j in range(1,6):
                if f'ses-0{j}' in base:
                    print(f'found ses-0{ses} in {base}')
                    ses = f'ses-0{j}'

            eventFileLong = f'{ROOT}/{sub}/ses-0{ses}/func/{base}_events.tsv'
            trEff = findTR(f'code/stimulation/{sub}/ses-01/{sub}_ses-01_run-01_neurovascularCoupling.log')
            trNom = trEff/4

    if longITIses == 2:
        skipLongITI = True
    else:
        skipLongITI = False

    # for modality in ['vaso', 'bold']:
    for modality in ['vaso']:
        print(f'Processing {modality}')

        if skipLongITI and len(allRuns) == 2:
            eventFileLong = f'{ROOT}/{sub}/ses-01/func/{sub}_ses-01_task-stimulation_run-01_part-mag_cbv_events.tsv'
            trEff = findTR(f'code/stimulation/{sub}/ses-01/{sub}_ses-01_run-01_neurovascularCoupling.log')
            trNom = trEff/4

        runs = sorted(glob.glob(f'{ROOT}/derivatives/{sub}/*/func/{sub}_ses-*_task-stimulation_run-avg_part-mag_{modality}_intemp.nii*'))

        designMatrices = []
        niis = []

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-21]
            print(f'Processing {base}')
            for i in range(1,10):
                if f'ses-0{i}' in base:
                    ses = f'ses-0{i}'

            trEff = findTR(f'code/stimulation/{sub}/{ses}/{sub}_{ses}_run-01_neurovascularCoupling.log')
            trNom = trEff/4

            nii = nb.load(run)
            nrTRs = nii.header['dim'][4]

            if (nrTRs > 240) and skipLongITI:
                print(f'Do not add this run. Will be added later')
                longBase = base
                continue

            else:
                print(f'Adding run')
                niis.append(nii)

            data = nii.get_fdata()
            nVols = data.shape[-1]
            frame_times = np.arange(nVols) * trNom

            events = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-01_part-mag_cbv_events.tsv', sep = ',')

            design_matrix = make_first_level_design_matrix(
                frame_times,
                events,
                hrf_model=hrf_model,
                drift_model = None,
                high_pass= high_pass
                )
            designMatrices.append(design_matrix)

        if skipLongITI:
            print(f'Adding average run')
            nii = nb.load(f'{funcDir}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_{modality}_intemp.nii.gz')

            niis.append(nii)

            data = nii.get_fdata()
            nVols = data.shape[-1]
            frame_times = np.arange(nVols) * trNom

            events = pd.read_csv('/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/sub-07/ses-03/func/sub-07_ses-03_task-stimulation_run-01_part-mag_cbv_events.tsv'
, sep = ',')

            design_matrix = make_first_level_design_matrix(
                frame_times,
                events,
                hrf_model=hrf_model,
                drift_model = None,
                high_pass= high_pass
                )




            designMatrices.append(design_matrix)

        # Set up contrasts
        contrast_matrix = np.eye(design_matrix.shape[1])
        basic_contrasts = dict([(column, contrast_matrix[i])
                    for i, column in enumerate(design_matrix.columns)])

        if modality == 'bold':
            contrasts = {'stimulation': + basic_contrasts['stim_1s']
                                        + basic_contrasts['stim_2s']
                                        + basic_contrasts['stim_4s']
                                        + basic_contrasts['stim_12s']
                                        + basic_contrasts['stim_24s'],

                        'stim_1s': + basic_contrasts['stim_1s'],
                        'stim_2s': + basic_contrasts['stim_2s'],
                        'stim_4s': + basic_contrasts['stim_4s'],
                        'stim_12s': + basic_contrasts['stim_12s'],
                        'stim_24s': + basic_contrasts['stim_24s']

                        }

        if modality == 'vaso':
            contrasts = {'stimulation': - basic_contrasts['stim_1s']
            - basic_contrasts['stim_2s']
            - basic_contrasts['stim_4s']
            - basic_contrasts['stim_12s']
            - basic_contrasts['stim_24s'],

            'stim_1s': - basic_contrasts['stim_1s'],
            'stim_2s': - basic_contrasts['stim_2s'],
            'stim_4s': - basic_contrasts['stim_4s'],
            'stim_12s': - basic_contrasts['stim_12s'],
            'stim_24s': - basic_contrasts['stim_24s']

            }

        # run GLM
        fmri_glm = FirstLevelModel(mask_img = False, drift_model=None)
        fmri_glm = fmri_glm.fit(niis, design_matrices = designMatrices)

        # Iterate on contrasts
        for contrast_id, contrast_val in contrasts.items():
            # compute the contrasts
            z_map = fmri_glm.compute_contrast(
                contrast_val, output_type='z_score')
            nb.save(z_map, f'{statFolder}/{sub}_{modality}_{contrast_id}.nii')

#
# files = glob.glob(f'{statFolder}/*.nii')
# for file in files:
#     base = file.split('.')[0]
#     command = f'fslmaths {file} -mul {ROOT}/derivatives/{sub}/{sub}_brainMask.nii.gz {base}_masked.nii.gz'
#     subprocess.run(command,shell=True)
#
#
# # Run GLM on ERA
# sub = ['sub-06']
#
# for sub in subs:
#
#     funcDir = f'{ROOT}/derivatives/{sub}'
#     # make folder to dump statistocal maps
#     statFolder = f'{funcDir}/statMaps'
#
#     for modality in ['vaso', 'bold']:
#
#         runs = sorted(glob.glob(f'{ROOT}/derivatives/{sub}/ERAs/{sub}_task-stimulation_run-avg_part-mag_{modality}_intemp_era-1s.nii.gz'))
#
#         designMatrices = []
#         niis = []
#
#         for run in runs:
#             base = os.path.basename(run).rsplit('.', 2)[0][:-21]
#
#             for i in range(1,10):
#                 if f'ses-0{i}' in base:
#                     ses = f'ses-0{i}'
#
#             # trEff = findTR(f'code/stimulation/{sub}/{ses}/{sub}_{ses}_run-01_neurovascularCoupling.log')
#             # trNom = trEff/4
#
#             # niiFile = f'{funcDir}/{base}_{modality}.nii.gz'
#             nii = nb.load(run)
#             niis.append(nii)
#
#             data = nii.get_fdata()
#             nVols = data.shape[-1]
#             frame_times = np.arange(nVols) * trNom
#
#             # events = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-01_part-mag_cbv_events.tsv', sep = ',')
#             events = pd.DataFrame({'onset': [0], 'duration': [2], 'trial_type': ['stim']})
#
#
#             design_matrix = make_first_level_design_matrix(
#                 frame_times,
#                 events,
#                 hrf_model=hrf_model,
#                 drift_model = None,
#                 high_pass= high_pass
#                 )
#             designMatrices.append(design_matrix)
#
#         contrast_matrix = np.eye(design_matrix.shape[1])
#         basic_contrasts = dict([(column, contrast_matrix[i])
#                     for i, column in enumerate(design_matrix.columns)])
#
#         if modality == 'bold':
#             contrasts = {'stimulation': + basic_contrasts['stim']}
#
#         if modality == 'vaso':
#             contrasts = {'stimulation': + basic_contrasts['stim']}
#
#         fmri_glm = FirstLevelModel(mask_img = False, drift_model=None)
#         fmri_glm = fmri_glm.fit(niis, design_matrices = designMatrices)
#
#         # Iterate on contrasts
#         for contrast_id, contrast_val in contrasts.items():
#             # compute the contrasts
#             z_map = fmri_glm.compute_contrast(
#                 contrast_val, output_type='z_score')
#             nb.save(z_map, f'{statFolder}/{sub}_{modality}_{contrast_id}_stim1s_test2.nii')
