"""Remove facial features from structural MRI data"""

import subprocess
import glob

# Set data path
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/openNeuro'

# Set subjects to work on
subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']

# Loop over participants
for sub in subs:
    images = sorted(glob.glob(f'{ROOT}/{sub}/*/anat/{sub}_ses-*_MP2RAGE.nii.gz'))

    for img in images:
        command = f'pydeface {img} '
        # print(command)
        subprocess.run(command, shell=True)
