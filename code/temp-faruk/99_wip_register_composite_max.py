'''Register composite max to MP2RAGE'''

import nibabel as nb
import subprocess
import os
import glob

CURRDIR = os.getcwd()
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'


TRANSFORM = f'{CURRDIR}/code/temp-faruk/sub-06_uni-to-megre_registered1_0GenericAffine.mat'
MOVINGFILE = f'{ROOT}/sub-06/ses-04/anat/megre/99_Faruk/sub-05_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_max.nii.gz'
FIXEDFILE = f'{ROOT}/sub-06/ses-01/anat/upsample/sub-06_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
OUTDIR = os.path.dirname(MOVINGDATA)

nii = nb.load(MOVINGDATA)
affine = nii.affine
data = nii.get_fdata()
nrEchoes = data.shape[-1]

# =============================================================================
# Split echoes
for echo in range(nrEchoes):
    tmp = data[...,echo]

    out_name = nii.get_filename().split(os.extsep, 1)[0]
    img = nb.Nifti1Image(tmp, affine=affine)
    nb.save(img, os.path.join(OUTDIR, "{}_echo{}.nii.gz".format(out_name,echo)))

# =============================================================================
# Register echoes
files = sorted(glob.glob(os.path.join(OUTDIR, "{}_echo*.nii.gz".format(out_name))))

for file in files:
    moving = f'{file}'
    basename, ext = moving.split(os.extsep, 1)

    command = 'antsApplyTransforms '
    command += f'--interpolation BSpline[5] '
    command += f'-d 3 -i {moving} '
    command += f'-r {FIXEDFILE} '
    command += f'-t [{TRANSFORM}, 1] '
    command += f'-o {basename}_registered.nii.gz'

    # Run command
    subprocess.run(command, shell = True)

# =============================================================================
# Merge echoes
OUTNAME = 'sub-05_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_max_registered.nii.gz'
outpath = os.path.join(OUTDIR, OUTNAME)
# Look for registered echoes
files = sorted(glob.glob(os.path.join(OUTDIR, "*_echo*registered.nii.gz")))

command1 = "fslmerge -t {}".format(outpath)

for f in files:
    command1 += " {}".format(f)

# Execute command
subprocess.run(command1, shell=True)
