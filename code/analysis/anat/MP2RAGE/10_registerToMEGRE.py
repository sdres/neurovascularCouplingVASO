"""Register each run to one reference run."""

import os
import subprocess
import nibabel as nb

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'
SUBS = ['sub-05']

fixed = f'{DATADIR}/{sub}/ses-01/anat/upsample/sub-05_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
moving = f"{DATADIR}/{sub}/{megreSes}/anat/megre/11_T2star/sub-05_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz"
regFolder = f"{DATADIR}/{sub}/ses-01/anat/07_register_to_T2s"
out_moving = f"{DATADIR}/{sub}/{megreSes}/anat/megre/11_T2star/sub-05_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0_registered.nii.gz"

initial = f'{DATADIR}/{sub}/{megreSes}/anat/megre/11_T2star/initial_matrix.txt'

command2 = "greedy "
command2 += "-d 3 "
command2 += "-rf {} ".format(moving)  # reference
command2 += "-ri LINEAR "
command2 += "-rm {} {} ".format(moving, out_moving)  # moving resliced
command2 += "-r {},-1 ".format(initial)
print("{}\n".format(command2))

# Execute command
subprocess.run(command2, shell=True)

for sub in SUBS:
    # Find MEGRE session
    for sesNr in range(1, 6):
        if os.path.exists(f"{DATADIR}/{sub}/ses-0{sesNr}/anat/megre/11_T2star/"
                          f"{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz"):
            megreSes = f'ses-0{sesNr}'

    # =============================================================================
    NII_NAMES = [
        f'{DATADIR}/{sub}/ses-01/anat/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
        # '{DATADIR}/{sub}/ses-01/anat/upsample/{sub}_LH_sphere_crop_ups4X.nii.gz'
        # f'{DATADIR}/{sub}/ses-01/anat/upsample/{sub}_rim-LH_perimeter_chunk_uncrop.nii.gz'
        # '{DATADIR}/{sub}/ses-01/anat/upsample/peri_uncrop.nii.gz'
        ]

    NII_TARGET = f"{DATADIR}/{sub}/{megreSes}/anat/megre/11_T2star/" \
                 f"{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz"

    # Use ITK-SNAP manually to find the best registration
    AFFINE = f"{DATADIR}/{sub}/{megreSes}/anat/megre/11_T2star/uni-to-megre_matrix.txt"
    # AFFINE = f"{DATADIR}/{sub}/{megreSes}/anat/megre/11_T2star/initial_matrix.txt"

    outDir = f"{DATADIR}/{sub}/ses-01/anat/07_register_to_T2s"

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

        # Substitute header with target nifti
        nii_target = nb.load(NII_TARGET)
        nii_moving = nb.load(out_moving)
        nii_out = nb.Nifti1Image(nii_moving.get_fdata(), header=nii_target.header,
                                 affine=nii_target.affine)
        nb.save(nii_out, out_moving)

        # crop vessels to sphere
        inFile = f'{basename}_registered.nii'
        base = inFile.split('.')[0]
        outFile = f'{base}_crop.nii.gz'

        command = 'fslroi '
        command += f'{inFile} '
        command += f'{outFile} '
        # command += '263 162 35 162 79 158'
        command += '271 162 7 162 31 159'

        subprocess.run(command, shell=True)

    print('\n\nFinished.')







# OLD SNIPPETS

#
#     moving = f'{DATADIR}/{sub}/ses-01/anat/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
#     # moving = f'{DATADIR}/{sub}/ses-01/anat/upsample/{sub}_rim-LH_perimeter_chunk.nii.gz'
#     fixed = f"{DATADIR}/{sub}/{megreSes}/anat/megre/11_T2star/{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz"
#     regFolder = f"{DATADIR}/{sub}/ses-01/anat/07_register_to_T2s"
#     initial = f"{DATADIR}/{sub}/{megreSes}/anat/megre/11_T2star/uni-to-megre_matrix.txt"
#
#     # Set up ants command
#     command = 'antsRegistration '
#     command += f'--verbose 1 '
#     command += f'--dimensionality 3 '
#     command += f'--float 0 '
#     command += f'--collapse-output-transforms 1 '
#     command += f'--interpolation BSpline[5] '
#     command += f'--output [{regFolder}/registered1_,{regFolder}/registered1_Warped.nii,1] '
#     command += f'--use-histogram-matching 0 '
#     command += f'--winsorize-image-intensities [0.005,0.995] '
#     command += f'--initial-moving-transform {initial} '
#     command += f'--transform Rigid[0.1,3,0] '
#     command += f'--metric CC[{fixed}, {moving},1,2] '
#     command += f'--convergence [60x10,1e-6,10] '
#     command += f'--shrink-factors 2x1 '
#     command += f'--smoothing-sigmas 1x0vox '
#     command += f'-x {DATADIR}/{sub}/{megreSes}/anat/megre/03_upsample/{sub}_ses-T2s_crop_regmask_ups2X.nii.gz'
#     # Run command
#     subprocess.run(command, shell=True)
#
#     moving = f'{DATADIR}/{sub}/ses-01/anat/upsample/{sub}_ses-01_inv-2_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
#
#     # Prepare command to apply transform and check quality
#     command = 'antsApplyTransforms '
#     # command += f'--interpolation BSpline[5] '
#     # command += f'--interpolation MultiLabel '
#     command += f'-d 3 -i {moving} '
#     command += f'-r {fixed} '
#     # command += f'-t {regFolder}/registered1_1Warp.nii.gz '
#     command += f'-t {regFolder}/registered1_0GenericAffine.mat '
#     command += f'-o {regFolder}/sub-05_rim-LH_perimeter_chunk_uncrop_registered.nii.gz'
#
#     # Run command
#     subprocess.run(command, shell=True)
#
#
# # Switch headers
# sub_06head = nb.load('/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-06/ses-04/anat/megre/11_T2star/sub-06_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz').header
#
# file = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-05/ses-02/anat/megre/11_T2star/sub-05_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz'
# sub_05_nii = nb.load(file)
# affine = sub_05_nii.affine
#
# new = nb.Nifti1Image(sub_05_nii.get_fdata(), header=sub_06head, affine=affine)
# nb.save(new, file)
#
#
#
#     # ==========================================================================
#     # Inverse registration to MP2RAGE
#     # ==========================================================================
#
#     # moving = f'{DATADIR}/sub-06/ses-04/anat/megre/11_T2star/sub-06_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_T2s.nii.gz'
#     moving = f'{DATADIR}/sub-08/ses-04/anat/megre/99_Faruk/sub-08_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_max.nii.gz'
#     basename, ext = moving.split(os.extsep, 1)
#     fixed = f'{DATADIR}/sub-08/ses-01/anat/upsample/sub-08_ses-01_inv-2_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
#     regFolder = f"{DATADIR}/sub-08/ses-01/anat/07_register_to_T2s"
#
#     # register vessels
#     command = 'antsApplyTransforms '
#     command += f'--interpolation BSpline[5] '
#     command += f'-d 3 -i {moving} '
#     command += f'-r {fixed} '
#     command += f'-t [{regFolder}/registered1_0GenericAffine.mat, 1] '
#     command += f'-o {basename}_registered.nii.gz'
#     # Run command
#     subprocess.run(command, shell = True)

#
#     # crop vessels to sphere
#     inFile = f'{basename}_registered.nii'
#     base = inFile.split('.')[0]
#     outFile = f'{base}_crop.nii.gz'
#
#     command = 'fslroi '
#     command += f'{inFile} '
#     command += f'{outFile} '
#     # command += '263 162 35 162 79 158'
#     command += '271 162 7 162 31 159'
#
#     subprocess.run(command, shell=True)
#
# # ==========================================================================
# # Inverse registration to MP2RAGE
# # ==========================================================================
#
# # moving = f'{DATADIR}/sub-06/ses-04/anat/megre/11_T2star/sub-06_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_T2s.nii.gz'
#
# moving = f'{DATADIR}/sub-05/ses-01/anat/megre/11_T2star/sub-05_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz'
# basename, ext = moving.split(os.extsep, 1)
# fixed = f'{DATADIR}/sub-05/ses-01/anat/upsample/sub-05_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
#
# regFolder = f"{DATADIR}/sub-05/ses-01/anat/07_register_to_T2s"
#
# # register vessels
# command = 'antsApplyTransforms '
# command += f'--interpolation BSpline[5] '
# command += f'-d 3 -i {moving} '
# command += f'-r {fixed} '
# command += f'-t [{regFolder}/uni-to-megre_matrix.txt, 1] '
# command += f'-o {basename}_registered.nii.gz'
# # Run command
# subprocess.run(command, shell=True)
#
# # crop vessels to sphere
# inFile = f'{basename}_registered.nii'
# base = inFile.split('.')[0]
# outFile = f'{base}_crop.nii.gz'
#
# command = 'fslroi '
# command += f'{inFile} '
# command += f'{outFile} '
# # command += '263 162 35 162 79 158'
# command += '271 162 7 162 31 159'
#
# subprocess.run(command, shell=True)
#
# # ======================================================================================================
# # Testing ANTs with switched affines
#
# # First test whether the matrix is the problem
# SUBS = ['sub-05']
#
# for sub in SUBS:
#     # Find MEGRE session
#     for sesNr in range(1, 6):
#         if os.path.exists(f"{DATADIR}/{sub}/ses-0{sesNr}/anat/megre/11_T2star/"
#                           f"{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz"):
#             megreSes = f'ses-0{sesNr}'
#
#     moving = f'{DATADIR}/{sub}/ses-01/anat/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
#     basename, ext = moving.split(os.extsep, 1)
#     # fixed = f'{DATADIR}/{sub}/{megreSes}/anat/megre/11_T2star/{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz'
#     fixed = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-05/ses-02/anat/megre/11_T2star/forcedAffine.nii'
#     regFolder = f"{DATADIR}/{sub}/ses-01/anat/07_register_to_T2s"
#     initial = f'{regFolder}/uni-to-megre_matrix.txt'
#
#     # Set up ants command
#     command = 'antsRegistration '
#     command += f'--verbose 1 '
#     command += f'--dimensionality 3 '
#     command += f'--float 0 '
#     command += f'--collapse-output-transforms 1 '
#     command += f'--interpolation BSpline[5] '
#     command += f'--output [{regFolder}/registered1_,{regFolder}/registered1_Warped.nii,1] '
#     command += f'--use-histogram-matching 0 '
#     command += f'--winsorize-image-intensities [0.005,0.995] '
#     command += f'--initial-moving-transform {initial} '
#     command += f'--transform Rigid[0.1,3,0] '
#     command += f'--metric CC[{fixed}, {moving},1,2] '
#     command += f'--convergence [60x10,1e-6,10] '
#     command += f'--shrink-factors 2x1 '
#     command += f'--smoothing-sigmas 1x0vox '
#     # command += f'-x {DATADIR}/{sub}/{megreSes}/anat/megre/03_upsample/{sub}_ses-T2s_crop_regmask_ups2X.nii.gz'
#     # Run command
#     subprocess.run(command, shell=True)
#
# # Force identity matrix into affine
# # of megre
# nii = nb.load(fixed)
# new = nb.Nifti1Image(nii.get_fdata(), affine=np.eye(4), header=nii.header)
# nb.save(new, '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-05/ses-02/anat/megre/11_T2star/forcedAffine.nii')
#
# import numpy as np
#
# # of mp2rage
# nii = nb.load(moving)
# new = nb.Nifti1Image(nii.get_fdata(), affine=np.eye(4), header=nii.header)
# nb.save(new, '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-05/ses-02/anat/megre/11_T2star/forcedAffine_mp2.nii')
#
#
# # Force affine matrix of original data
# nii = nb.load('/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-05/ses-02/anat/megre/11_T2star/sub-05_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz')
# orig = nb.load('/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/sub-05/ses-02/anat/sub-05_ses-02_T2s_run-01_dir-AP_echo-1_part-phase_ASPMEGRE.nii.gz')
#
# new = nb.Nifti1Image(nii.get_fdata(), affine=orig.affine, header=orig.header)
# nb.save(new, '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-05/ses-02/anat/megre/11_T2star/forcedAffine.nii.gz')
