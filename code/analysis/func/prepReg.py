"""

Bypass the required registration step for higher level analyses in FSL FEAT

"""

import subprocess
import glob

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

subs = ['sub-06']

for sub in subs:
    featFolders = sorted(glob.glob(f'{ROOT}/derivatives/{sub}/ses-0*/func/*vaso*_trunc.feat'))
    len(featFolders)
    for folder in featFolders:
        # Make registration folder
        command = f'mkdir {folder}/reg'
        subprocess.run(command, shell=True)
        # Copy identity matrix to registration folder
        command = f'cp $FSLDIR/etc/flirtsch/ident.mat {folder}/reg/example_func2standard.mat'
        subprocess.run(command, shell=True)
        # Do something because Mumford brain stats does it too
        command = f'cp {folder}/mean_func.nii.gz {folder}/reg/standard.nii.gz'
        subprocess.run(command, shell=True)
