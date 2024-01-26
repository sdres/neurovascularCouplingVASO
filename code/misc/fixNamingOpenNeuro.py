"""Script to fix the naming of files for upload on openneuro"""

import shutil
import glob

FOLDER = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/openNeuro/'

# Find files
files = glob.glob(f'{FOLDER}/sub-*/ses-*/anat/*MEGRE*')

template = f'{}'


shutil.copyfile(src, dst)
