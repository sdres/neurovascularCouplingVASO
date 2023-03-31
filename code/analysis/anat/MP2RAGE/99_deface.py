"""Remove facial features from structural MRI data"""

import subprocess
import glob

# Set data path
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# Set subjects to work on
subs = ['sub-09']

# Loop over participants
for sub in subs:

    images = sorted(glob.glob(f'{ROOT}/{sub}/*/anat/{sub}_ses-*_MP2RAGE.nii.gz'))

    for img in images:
        command = f'pydeface {img}'
        subprocess.run(command, shell=True)
