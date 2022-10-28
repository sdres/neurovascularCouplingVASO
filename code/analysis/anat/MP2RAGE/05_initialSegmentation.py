'''

Running intitial segmentations in FSL FAST on upsampled UNI image cropped to sphere.

'''
import subprocess


SUBS = ['sub-07']
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

for sub in SUBS:

    command = 'fast '
    command += '-t 1 -n 3 -H 0.1 -I 4 -l 20.0 '
    command += f'-o {DATADIR}/{sub}/segmentation/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X_crop-toSphere '
    command += f'{DATADIR}/{sub}/ses-01/anat/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X_crop-toSphere'

    subprocess.run(command, shell = True)
