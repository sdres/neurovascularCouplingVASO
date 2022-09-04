''' Get event-related avergaes per stimulus duration '''

import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import pandas as pd
import re
import matplotlib.pyplot as plt

# Define root dir
ROOT = '/Users/sebastiandresbach/git/neurovascularCouplingVASO'
# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-01'
# Define subjects to work on
subs = ['sub-01']

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

    # Find mean trigger-time
    meanTriggerDur = (meanFirstTriggerDur+meanSecondTriggerDur)/2
    return meanTriggerDur



# =============================================================================
# Extract timecourses
# =============================================================================

for sub in subs:
    for modality in ['vaso', 'bold']:
        run = f'{DATADIR}/{sub}_task-stimulation_part-mag_{modality}_intemp.nii.gz'

        nii = nb.load(run)
        header = nii.header
        data = nii.get_fdata()

        mask = nb.load(f'{DATADIR}/v1Mask.nii.gz').get_fdata()

        mask_mean = np.mean(data[:, :, :][mask.astype(bool)],axis=0)

        np.save(f'{DATADIR}/{sub}_task-stimulation_part-mag_{modality}_intemp_timecourse', mask_mean)



# =============================================================================
# Upsample timecourses
# =============================================================================

from scipy import interpolate

logFile = f'../stimulation/{sub}/ses-01/{sub}_ses-01_run-01_neurovascularCoupling.log'
tr = findTR(logFile)

jitter = 6
factor = jitter/2

for sub in subs:
    for modality in ['vaso', 'bold']:
        timecourse = np.load(f'{DATADIR}/{sub}_task-stimulation_part-mag_{modality}_intemp_timecourse.npy')

        x = np.arange(0, timecourse.shape[0])
        f = interpolate.interp1d(x, timecourse, fill_value='extrapolate', kind = 'cubic')

        xNew = np.arange(0, timecourse.shape[0], 1/factor)
        new = f(xNew)

        # Check whether interoplation makes sense
        # plt.figure()
        # plt.plot(x[:10], timecourse[:10], 'x')
        # plt.plot(xNew[:30], new[:30], 'o', alpha=0.5)
        # plt.show()

        # Save new timecourse
        np.save(f'{DATADIR}/{sub}_task-stimulation_part-mag_{modality}_intemp_timecourse_intemp', new)
        
