"""Register each run to one reference run."""

import os
import subprocess
import numpy as np
import nibabel as nb

# =============================================================================
NII_NAMES = [
    '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/anat/upsample/sub-06_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
    # '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/anat/upsample/sub-06_LH_sphere_crop_ups4X.nii.gz'
    # '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/anat/upsample/peri_uncrop.nii.gz'
    ]
NII_TARGET = "/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/11_T2star/sub-06_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz"

# Use ITK-SNAP manually to find the best registration
AFFINE = "/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/11_T2star/initial_matrix.txt"

outDir = "/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/anat/07_register_to_T2s"

# =============================================================================
print("Step_07: Apply registration from T1 to T2s space")

# Output directory
if not os.path.exists(outDir):
    os.makedirs(outDir)
print("  Output directory: {}\n".format(outDir))

# -------------------------------------------------------------------------
# Apply affine transformation matrix
# -------------------------------------------------------------------------
for i in range(0, len(NII_NAMES)):
    # Prepare inputs
    in_moving = NII_NAMES[i]
    in_affine = AFFINE

    # Prepare output
    basename, ext = in_moving.split(os.extsep, 1)
    basename = os.path.basename(basename)
    print(basename)
    out_moving = os.path.join(outDir, "{}_reg.nii.gz".format(basename))
    out_affine = os.path.join(outDir, "{}_affine.mat".format(basename))

    command2 = "greedy "
    command2 += "-d 3 "
    command2 += "-rf {} ".format(NII_TARGET)  # reference
    command2 += "-ri LINEAR "
    command2 += "-rm {} {} ".format(in_moving, out_moving)  # moving resliced
    command2 += "-o {} ".format(out_affine)
    command2 += "-r {} ".format(in_affine)
    print("{}\n".format(command2))

    # Execute command
    subprocess.run(command2, shell=True)

    # Substitude header with target nifti
    nii_target = nb.load(NII_TARGET)
    nii_moving = nb.load(out_moving)
    nii_out = nb.Nifti1Image(nii_moving.get_fdata(), header=nii_target.header,
                             affine=nii_target.affine)
    nb.save(nii_out, out_moving)

print('\n\nFinished.')



moving = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/anat/upsample/sub-06_ses-01_inv-2_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
fixed = "/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/11_T2star/sub-06_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz"
regFolder = "/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/anat/07_register_to_T2s"
initial = "/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/11_T2star/initial_matrix.txt"

# Set up ants command
command = 'antsRegistration '
command += f'--verbose 1 '
command += f'--dimensionality 3 '
command += f'--float 0 '
command += f'--collapse-output-transforms 1 '
command += f'--interpolation BSpline[5] '
command += f'--output [{regFolder}/registered1_,{regFolder}/registered1_Warped.nii,1] '
command += f'--use-histogram-matching 0 '
command += f'--winsorize-image-intensities [0.005,0.995] '
command += f'--initial-moving-transform {initial} '
command += f'--transform Rigid[0.1,3,0] '
command += f'--metric MI[{fixed}, {moving},1,2] '
command += f'--convergence [60x10,1e-6,10] '
command += f'--shrink-factors 2x1 '
command += f'--smoothing-sigmas 1x0vox '
command += f'-x /Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/03_upsample/sub-06_ses-T2s_crop_regmask_ups2X.nii.gz'
# Run command
subprocess.run(command,shell=True)


# Prepare command to apply transform and check quality
command = 'antsApplyTransforms '
# command += f'--interpolation BSpline[5] '
command += f'--interpolation MultiLabel '
command += f'-d 3 -i {moving} '
command += f'-r {fixed} '
# command += f'-t {regFolder}/registered1_1Warp.nii.gz '
command += f'-t {regFolder}/registered1_0GenericAffine.mat '
command += f'-o {regFolder}/peri_registered.nii'
# Run command
subprocess.run(command,shell=True)

# moving = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/anat/upsample/peri_uncrop.nii.gz'
moving = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/finalVeins.nii.gz'
basename, ext = moving.split(os.extsep, 1)
fixed = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-01/anat/upsample/sub-06_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'

# register vessels
command = 'antsApplyTransforms '
command += f'--interpolation MultiLabel '
command += f'-d 3 -i {moving} '
command += f'-r {fixed} '
# command += f'-t {regFolder}/registered1_1Warp.nii.gz '
command += f'-t [{regFolder}/registered1_0GenericAffine.mat, 1] '
command += f'-o {basename}_registered.nii'
# Run command
subprocess.run(command,shell=True)


# crop vessels to sphere
inFile = f'{basename}_registered.nii'
base = inFile.split('.')[0]
outFile = f'{base}_crop.nii.gz'

command = 'fslroi '
command += f'{inFile} '
command += f'{outFile} '
# command += '263 162 35 162 79 158'
command += '271 162 7 162 31 159'

subprocess.run(command,shell=True)
