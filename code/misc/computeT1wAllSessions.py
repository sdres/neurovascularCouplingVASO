'''
Computes T1w image in EPI-space from motion-corrected 'nulled' and 'notnulled'
timeseries of all sessions as acquired with SS-SI VASO.
'''

import numpy as np
import nibabel as nb
from scipy import signal
import glob

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

nulledFiles = sorted(glob.glob(f'{ROOT}/sub-08/ses-0*/func/sub-08_ses-0*_task-stimulation_run-avg_part-mag_cbv.nii'))
notnulledFiles = sorted(glob.glob(f'{ROOT}/sub-08/ses-0*/func/sub-08_ses-0*_task-stimulation_run-avg_part-mag_bold.nii'))

test = computeT1w(nulledFiles,notnulledFiles)
# Get header and affine
header = nb.load(nulledFiles[0]).header
affine = nb.load(nulledFiles[0]).affine

# And save the image
img = nb.Nifti1Image(test, header = header, affine = affine)
nb.save(img, f'{ROOT}/sub-08/sub-08_ses-avg_task-stimulation_run-avg_part-mag_T1w.nii')


def computeT1w(nulledFiles, notnulledFiles, detrend = False):

    '''
    Takes nulled and notnulled files as lists as input and computes T1w image
    in EPI space. Returns array instead of saving a file to allow different
    naming conventions.
    '''

    for i, (nulledFile, notnulledFile) in enumerate(zip(nulledFiles,notnulledFiles)):
        # Load nulled motion corrected timeseries
        nulledNii = nb.load(nulledFile)
        nulledData = nulledNii.get_fdata()

        # Load notnulled motion corrected timeseries
        notnulledNii = nb.load(notnulledFile)
        notnulledData = notnulledNii.get_fdata()

        if i == 0:

            # Concatenate nulled and notnulled timeseries
            combined = np.concatenate((notnulledData,nulledData), axis=3)

        else:
            combined = np.concatenate((combined,nulledData), axis=3)
            combined = np.concatenate((combined,notnulledData), axis=3)


    if detrend == True:
        # Detrend before std. dev. calculation
        combinedDemean = signal.detrend(combined, axis = 3, type = 'constant')
        combinedDetrend = signal.detrend(combinedDemean, axis = 3, type = 'linear')
        stdDev = np.std(combinedDetrend, axis = 3)
    else:
        stdDev = np.std(combined, axis = 3)

    #Compute mean
    mean = np.mean(combined, axis = 3)
    # Compute variation
    cvar = stdDev/mean
    # Take inverse
    cvarInv = 1/cvar

    return cvarInv
