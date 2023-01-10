"""Merge event related average measurements into a single nifti."""

import os
import subprocess
import numpy as np
import nibabel as nb
import glob

# =============================================================================
FOLDER = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ERAs/frames'
NII_NAMES = sorted(glob.glob(f'{FOLDER}/sub-06_task-stimulation_run-avg_part-mag_vaso_intemp_era-24s_sigChange_masked_frame*_registered_crop.nii.gz'))

NII_NAMES = [
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame00_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame01_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame02_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame03_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame04_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame05_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame06_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame07_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame08_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame09_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame10_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame11_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame12_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame13_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame14_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame15_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame16_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame17_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame18_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame19_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame20_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame21_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame22_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame23_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame24_registered_crop.nii.gz',
    '/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs/sub-06_task-stimulation_run-avg_part-mag_bold_intemp_era-2s_sigChange_masked_frame25_registered_crop.nii.gz',
    ]

OUTDIR = "/home/faruk/data2/temp_OHBM_2023_1/cropped/ERAs_4D"
OUTDIR = FOLDER
OUTNAME = "sub-06_task-stimulation_run-avg_part-mag_vaso_intemp_era-24s_sigChange_masked_registered_crop.nii.gz"

# =============================================================================
# Output directory
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
print("  Output directory: {}\n".format(OUTDIR))

# Prepare command
outpath = os.path.join(OUTDIR, OUTNAME)
command1 = "fslmerge -t {}".format(outpath)

for f in NII_NAMES:
    command1 += " {}".format(f)

# Execute command
print(command1)
subprocess.run(command1, shell=True)

print('\n\nFinished.')
