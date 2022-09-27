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
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

subs = ['sub-03']

drift_model = 'Cosine'  # We use a discrete cosine transform to model signal drifts.
high_pass = .01  # The cutoff for the drift model is 0.01 Hz.
hrf_model = 'spm'  # The hemodynamic response function is the SPM canonical one.


def findTR(logfile):
    with open(logfile) as f:
        f = f.readlines()

    triggerTimes = []
    for line in f[1:]:
        if re.findall("Keypress: 5",line):
            triggerTimes.append(float(re.findall("\d+\.\d+", line)[0]))

    triggerTimes[0] = 0

    triggersSubtracted = []
    for n in range(len(triggerTimes)-1):
        triggersSubtracted.append(float(triggerTimes[n+1])-float(triggerTimes[n]))

    meanFirstTriggerDur = np.mean(triggersSubtracted[::2])
    meanSecondTriggerDur = np.mean(triggersSubtracted[1::2])

    # find mean trigger-time
    meanTriggerDur = (meanFirstTriggerDur+meanSecondTriggerDur)/2
    return meanTriggerDur



for sub in subs:

    funcDir = f'{ROOT}/derivatives/{sub}/func'
    # make folder to dump statistocal maps
    statFolder = f'{funcDir}/statMaps'
    if not os.path.exists(statFolder):
        os.makedirs(statFolder)
        print("Statmap directory is created")

    for ses in ['ses-01']:


        for acquiType in ['SingleShot', 'MultiShot']:

            if acquiType == 'SingleShot':
                tr = findTR(f'code/stimulation/{sub}/ses-01/{sub}ses-01blockStim_30sOnOffrun-01.log')
            if acquiType == 'MultiShot':
                tr = findTR(f'code/stimulation/{sub}/ses-01/{sub}ses-01blockStim_30sOnOffrun-03.log')
            print(f'{acquiType} tr: {tr}')

        for modality in ['vaso', 'bold']:

            runs = sorted(glob.glob(f'{ROOT}/derivatives/{sub}/{sub}_task-stim{acquiType}_*_{modality}_intemp.nii*'))

            for run in runs:
                base = os.path.basename(run).rsplit('.', 2)[0][:-21]

                # niiFile = f'{funcDir}/{base}_{modality}.nii.gz'
                nii = nb.load(run)
                data = nii.get_fdata()
                nVols = data.shape[-1]
                frame_times = np.arange(nVols) * tr

                events = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-stim{acquiType}_run-01_part-mag_events.tsv', sep = ',')

                design_matrix = make_first_level_design_matrix(
                    frame_times,
                    events,
                    hrf_model=hrf_model,
                    drift_model = None,
                    high_pass= high_pass
                    )

                contrast_matrix = np.eye(design_matrix.shape[1])
                basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])

                if modality == 'bold':
                    contrasts = {'stimulation': + basic_contrasts['stimulation']
                        }

                if modality == 'vaso':
                    contrasts = {'stimulation': - basic_contrasts['stimulation']
                        }
                        
                fmri_glm = FirstLevelModel(mask_img = False, drift_model=None)
                fmri_glm = fmri_glm.fit(nii, design_matrices = design_matrix)

                # Iterate on contrasts
                for contrast_id, contrast_val in contrasts.items():
                    # compute the contrasts
                    z_map = fmri_glm.compute_contrast(
                        contrast_val, output_type='z_score')
                    nb.save(z_map, f'{statFolder}/{base}_{modality}_{contrast_id}.nii')
