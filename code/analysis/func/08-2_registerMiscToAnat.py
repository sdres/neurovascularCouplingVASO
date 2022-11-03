
import glob
import os
import subprocess
import nibabel as nb

subs = ['sub-05']

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'


for sub in subs:

    # Defining folders
    mapDir = f'{DATADIR}/{sub}/statMaps'  # Location of functional data
    anatDir = f'{DATADIR}/{sub}/ses-01/anat'  # Location of anatomical data
    regFolder = f'{anatDir}/registrationFiles'  # Folder where output will be saved


    # register maps
    statMaps = sorted(glob.glob(f'{mapDir}/*.nii'))

    for statMap in statMaps:
        # =========================================================================
        # Apply inverse transform
        # =========================================================================

        # Take care: fixed and moving are flipped
        fixed = glob.glob(f'{anatDir}/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz')[0]

        moving = statMap

        command = 'antsApplyTransforms '
        command += f'--interpolation BSpline[5] '
        command += f'-d 3 '
        command += f'-i {moving} '
        command += f'-r {fixed} '
        command += f'-t {regFolder}/registered1_1InverseWarp.nii.gz '
        # IMPORTANT: We take the inverse transform!!!
        command += f'-t [{regFolder}/registered1_0GenericAffine.mat, 1] '
        command += f'-o {moving.split(".")[0]}_registered.nii.gz'

        subprocess.run(command,shell=True)

        command = f'fslmaths {moving.split(".")[0]}_registered.nii.gz -mul 1 {moving.split(".")[0]}_registered.nii.gz -odt float'
        subprocess.run(command,shell=True)

        # =========================================================================
        # Crop map
        # =========================================================================

        inFile = f'{moving.split(".")[0]}_registered.nii.gz'
        base = inFile.split('.')[0]
        outFile = f'{base}_crop.nii.gz'

        command = 'fslroi '
        command += f'{inFile} '
        command += f'{outFile} '
        command += '263 162 35 162 79 158'
        # command += '271 162 7 162 31 159'

        subprocess.run(command,shell=True)


    # register timeseries
    eraDir = f'{DATADIR}/{sub}/ERAs'  # Location of functional data
    eras = sorted(glob.glob(f'{eraDir}/*masked.nii.gz'))
    outFolder = f'{eraDir}/frames'

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
            frame = data[...,i]
            img = nb.Nifti1Image(frame, header=header,affine=affine)
            nb.save(img, outName)

            # ==================================================================
            # Mask with sphere
            command = 'fslmaths '
            command += f'{outName} '
            command += f'-mul {anatDir}/upsample/{sub}_LH_sphere_ups4X_registered.nii '
            command += f'{outName}'

            subprocess.run(command,shell=True)

            # ==================================================================
            # Apply inverse transform
            # Take care: fixed and moving are flipped
            fixed = glob.glob(f'{anatDir}/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz')[0]

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

            subprocess.run(command,shell=True)

            command = f'fslmaths {moving.split(".")[0]}_registered.nii.gz -mul 1 {moving.split(".")[0]}_registered.nii.gz -odt float'
            subprocess.run(command,shell=True)

            # =========================================================================
            # Crop map
            inFile = f'{moving.split(".")[0]}_registered.nii.gz'
            base = inFile.split('.')[0]
            outFile = f'{base}_crop.nii.gz'

            command = 'fslroi '
            command += f'{inFile} '
            command += f'{outFile} '
            command += '263 162 35 162 79 158'
            # command += '271 162 7 162 31 159'

            subprocess.run(command,shell=True)


            # Delete large registered file
            command = f'rm {inFile}'
            subprocess.run(command, shell = True)
