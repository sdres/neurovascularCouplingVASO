'''

Removing phase images of timeseries to save space

'''

import subprocess
import glob


FOLDER = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
files = sorted(glob.glob(f'{FOLDER}/sub-*/ses-*/func/*_part-phase*'))

for file in files:
    command = f'rm {file}'
    subprocess.run(command, shell = True)
