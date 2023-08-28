
import glob
import os
import subprocess
import nibabel as nb

subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
subs = ['sub-08']

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

BBOX = {'sub-05': {'RH': {'xlower': 435, 'xrange': 162, 'ylower': 55, 'yrange': 162, 'zlower': 95, 'zrange': 158},
                   'LH': {'xlower': 263, 'xrange': 162, 'ylower': 35, 'yrange': 162, 'zlower': 79, 'zrange': 158}},
        'sub-06': {'LH': {'xlower': 271, 'xrange': 162, 'ylower': 7, 'yrange': 162, 'zlower': 31, 'zrange': 159}},
        'sub-07': {'LH': {'xlower': 271, 'xrange': 166, 'ylower': 35, 'yrange': 158, 'zlower': 23, 'zrange': 166}},
        'sub-08': {'LH': {'xlower': 275, 'xrange': 162, 'ylower': 15, 'yrange': 162, 'zlower': 47, 'zrange': 158}},
        'sub-09': {'RH': {'xlower': 415, 'xrange': 162, 'ylower': 11, 'yrange': 162, 'zlower': 91, 'zrange': 158},
                   'LH': {'xlower': 303, 'xrange': 162, 'ylower': 0, 'yrange': 162, 'zlower': 59, 'zrange': 158}}
        }

for sub in subs:

    print(f'Processing {sub}')
    # Defining folders
    anatDir = f'{DATADIR}/{sub}/ses-01/anat'  # Location of anatomical data
    regFolder = f'{anatDir}/registrationFiles'  # Folder where output will be saved
    #
    # # =========================================================================
    # # register QA
    # for measure in ['mean', 'skew', 'kurt']:
    #
    #     subDir = f'{DATADIR}/{sub}/'  # Location of statistical maps
    #     qaMaps = sorted(glob.glob(f'{subDir}/ses-0*/func/*avg*{measure}.nii*'))
    #
    #     for qaMap in qaMaps:
    #
    #         print(f'Processing {qaMap}')
    #         # =========================================================================
    #         # Apply inverse transform
    #         # =========================================================================
    #
    #         # Take care: fixed and moving are flipped
    #         fixed = glob.glob(f'{anatDir}/upsample/'
    #                           f'{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz')[0]
    #
    #         moving = qaMap
    #
    #         command = 'antsApplyTransforms '
    #         command += f'--interpolation BSpline[5] '
    #         command += f'-d 3 '
    #         command += f'-i {moving} '
    #         command += f'-r {fixed} '
    #         command += f'-t {regFolder}/registered1_1InverseWarp.nii.gz '
    #         # IMPORTANT: We take the inverse transform!!!
    #         command += f'-t [{regFolder}/registered1_0GenericAffine.mat, 1] '
    #         command += f'-o {moving.split(".")[0]}_registered.nii.gz'
    #
    #         subprocess.run(command, shell=True)
    #
    #         command = f'fslmaths ' \
    #                   f'{moving.split(".")[0]}_registered.nii.gz ' \
    #                   f'-mul 1 ' \
    #                   f'{moving.split(".")[0]}_registered.nii.gz ' \
    #                   f'-odt float'
    #
    #         subprocess.run(command, shell=True)
    #
    #         # =========================================================================
    #         # Crop map
    #
    #         inFile = f'{moving.split(".")[0]}_registered.nii.gz'
    #
    #         base = inFile.split('.')[0]
    #         outFile = f'{base}_crop.nii.gz'
    #
    #         for hemi in BBOX[sub]:
    #             tmpBox = BBOX[sub][hemi]
    #             outFile = f'{base}_crop-toShpere{hemi}.nii.gz'
    #
    #             command = 'fslroi '
    #             command += f'{inFile} '
    #             command += f'{outFile} '
    #             command += f"{tmpBox['xlower']} " \
    #                        f"{tmpBox['xrange']} " \
    #                        f"{tmpBox['ylower']} " \
    #                        f"{tmpBox['yrange']} " \
    #                        f"{tmpBox['zlower']} " \
    #                        f"{tmpBox['zrange']}"
    #
    #             subprocess.run(command, shell=True)
    #
    # # =========================================================================
    # # register maps
    # for analysisPackage in ['fsl']:
    #
    #     mapDir = f'{DATADIR}/{sub}/statMaps/glm_{analysisPackage}'  # Location of statistical maps
    #     statMaps = sorted(glob.glob(f'{mapDir}/*cope.nii*'))
    #
    #     for statMap in statMaps:
    #
    #         print(f'Processing {statMap}')
    #         # =========================================================================
    #         # Apply inverse transform
    #         # =========================================================================
    #
    #         # Take care: fixed and moving are flipped
    #         fixed = glob.glob(f'{anatDir}/upsample/'
    #                           f'{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz')[0]
    #
    #         moving = statMap
    #
    #         command = 'antsApplyTransforms '
    #         command += f'--interpolation BSpline[5] '
    #         command += f'-d 3 '
    #         command += f'-i {moving} '
    #         command += f'-r {fixed} '
    #         command += f'-t {regFolder}/registered1_1InverseWarp.nii.gz '
    #         # IMPORTANT: We take the inverse transform!!!
    #         command += f'-t [{regFolder}/registered1_0GenericAffine.mat, 1] '
    #         command += f'-o {moving.split(".")[0]}_registered.nii.gz'
    #
    #         # subprocess.run(command, shell=True)
    #
    #         command = f'fslmaths ' \
    #                   f'{moving.split(".")[0]}_registered.nii.gz ' \
    #                   f'-mul 1 ' \
    #                   f'{moving.split(".")[0]}_registered.nii.gz ' \
    #                   f'-odt float'
    #
    #         # subprocess.run(command, shell=True)
    #
    #         # =========================================================================
    #         # Crop map
    #
    #         inFile = f'{moving.split(".")[0]}_registered.nii.gz'
    #
    #         base = inFile.split('.')[0]
    #         outFile = f'{base}_crop.nii.gz'
    #
    #         for hemi in BBOX[sub]:
    #             tmpBox = BBOX[sub][hemi]
    #             outFile = f'{base}_crop-toShpere{hemi}.nii.gz'
    #
    #             command = 'fslroi '
    #             command += f'{inFile} '
    #             command += f'{outFile} '
    #             command += f"{tmpBox['xlower']} " \
    #                        f"{tmpBox['xrange']} " \
    #                        f"{tmpBox['ylower']} " \
    #                        f"{tmpBox['yrange']} " \
    #                        f"{tmpBox['zlower']} " \
    #                        f"{tmpBox['zrange']}"
    #
    #             # subprocess.run(command, shell=True)

    # =========================================================================
    # register timeseries
    # =========================================================================

    eraDir = f'{DATADIR}/{sub}/ERAs'  # Location of functional data
    eras = sorted(glob.glob(f'{eraDir}/*ses-avg*after.nii.gz'))
    outFolder = f'{eraDir}/frames'
    tmpBox = BBOX[sub]['LH']

    # Make output folder if it does not exist already
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    for era in eras:

        basename = os.path.basename(era).split('.')[0]

        # Split frames
        nii = nb.load(era)
        header = nii.header
        affine = nii.affine
        data = nii.get_fdata()

        for i in range(data.shape[-1]):
            outName = f'{outFolder}/{basename}_frame{i:02d}.nii.gz'

            if os.path.exists(outName):
                print(f'file exists')
                continue

            frame = data[..., i]
            img = nb.Nifti1Image(frame, header=header, affine=affine)
            nb.save(img, outName)

            # # ==================================================================
            # # Mask with sphere
            # command = 'fslmaths '
            # command += f'{outName} '
            # command += f'-mul {anatDir}/upsample/{sub}_LH_sphere_ups4X_registered.nii '
            # command += f'{outName}'
            #
            # subprocess.run(command, shell=True)

            # ==================================================================
            # Apply inverse transform
            # Take care: fixed and moving are flipped
            fixed = glob.glob(f'{anatDir}/upsample/'
                              f'{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz')[0]

            moving = outName

            command = 'antsApplyTransforms '
            command += f'--interpolation BSpline[5] '
            command += f'-d 3 '
            command += f'-i {moving} '
            command += f'-r {fixed} '
            command += f'-t {regFolder}/registered1_1InverseWarp.nii.gz '
            # IMPORTANT: We take the inverse transform!!!
            command += f'-t [{regFolder}/registered1_0GenericAffine.mat, 1] '
            command += f'-o {moving.split(".")[0]}_registered.nii.gz'

            subprocess.run(command, shell=True)

            command = f'fslmaths {moving.split(".")[0]}_registered.nii.gz ' \
                      f'-mul 1 ' \
                      f'{moving.split(".")[0]}_registered.nii.gz ' \
                      f'-odt float'

            subprocess.run(command, shell=True)

            # =========================================================================
            # Crop map
            inFile = f'{moving.split(".")[0]}_registered.nii.gz'
            base = inFile.split('.')[0]
            outFile = f'{base}_crop.nii.gz'

            command = 'fslroi '
            command += f'{inFile} '
            command += f'{outFile} '
            command += f"{tmpBox['xlower']} " \
                       f"{tmpBox['xrange']} " \
                       f"{tmpBox['ylower']} " \
                       f"{tmpBox['yrange']} " \
                       f"{tmpBox['zlower']} " \
                       f"{tmpBox['zrange']}"

            subprocess.run(command, shell=True)

            # =========================================================================
            # Delete large registered file
            command = f'rm {inFile}'
            subprocess.run(command, shell=True)
