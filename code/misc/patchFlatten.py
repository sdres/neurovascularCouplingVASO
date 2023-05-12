"""Automative patch flattening of all participants"""

import subprocess
import os
import glob

subs = ['sub-06']

for sub in subs:
    subFolder = f'/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/{sub}'
    anatFolder = f'{subFolder}/ses-01/anat/upsample'

    # Make output directory
    outDir = f'{subFolder}/patchFlatten'
    # Make output folder if it does not exist already
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    files = [f'{anatFolder}/{sub}_rim-LH_curvature_binned.nii.gz']

    statMaps = sorted(glob.glob(f'{subFolder}/statMaps/glm_fsl/*toShpere*'))
    for file in statMaps:
        files.append(file)

    # Find MEGRE session
    NII_NAMES = sorted(glob.glob(f'{subFolder}/ses-*/anat/megre/01_crop/'
                                 f'{sub}_ses-*_T2s_run-01_dir-AP_echo-5_part-mag_MEGRE_crop.nii.gz'))
    for i in range(1, 6):  # We had a maximum of 5 sessions
        if f'ses-0{i}' in NII_NAMES[0]:
            ses = f'ses-0{i}'
    # Add T2* image to files
    t2star = f'{subFolder}/{ses}/anat/megre/11_T2star/{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_T2s_registered_crop-toShpereLH.nii.gz'
    files.append(t2star)

    for file in files:
        base = file.split('/')[-1].split('.')[0]
        print(f'Processing {base}')

        command1 = 'LN2_PATCH_FLATTEN ' \
                   f'-values {file} ' \
                   f'-coord_uv {anatFolder}/{sub}_rim-LH_UV_coordinates.nii.gz ' \
                   f'-coord_d {anatFolder}/{sub}_rim-LH_metric_equivol.nii.gz ' \
                   f'-domain {anatFolder}/{sub}_rim-LH_perimeter_chunk.nii.gz ' \
                   '-voronoi ' \
                   '-bins_u 1000 ' \
                   '-bins_v 1000 ' \
                   '-bins_d 100 ' \
                   f'-output {outDir}/{base}'

        # subprocess.run(command1, shell=True)

        # Fix header information
        command2 = f'fslmaths {outDir}/sub-06_rim-LH_curvature_binned_flat_1000x1000_voronoi.nii ' \
                   f'-mul 0 ' \
                   f'-add {outDir}/{base}_flat_1000x1000_voronoi.nii ' \
                   f'{outDir}/{base}_flat_1000x1000_voronoi'

        # subprocess.run(command2, shell=True)

        # Remove uncompressed
        command3 = f'rm {outDir}/{base}_flat_1000x1000_voronoi.nii'
        subprocess.run(command3, shell=True)
