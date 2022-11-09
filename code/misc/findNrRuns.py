'''

Find the number of individual runs per participant

'''

import glob

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

SUBS = ['sub-05', 'sub-06', 'sub-07','sub-08']

dict = {}

for sub in SUBS:
    runs = glob.glob(f'{ROOT}/{sub}/ses-*/func/*cbv.nii.gz')
    dict[sub] = len(runs)

dict
