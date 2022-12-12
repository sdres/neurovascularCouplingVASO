'''

Doing LayNii stuff like:
- layerification
- Defining perimeter
- Flattening

'''

import subprocess
import nibabel as nb
import numpy as np

# Set data path
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# Set subjects to work on
subs = ['sub-09']
# Set sessions to work on
# sessions = ['ses-01']
#
# # Defining control points for all subjects
# controlPoints = {'sub-05': [281, 209, 120],
#                  'sub-06': [258, 169, 85],
#                  'sub-07': [275, 206, 100],
#                  'sub-10': [335, 174, 105]}
# radii = {'sub-05': 11,
#          'sub-06': 11,
#          'sub-07': 11,
#          'sub-09': 11,
#          'sub-10': 11}


for sub in subs:
    subDir = f'{DATADIR}/{sub}'

    # Set rim file
    rimFile = f'{subDir}/ses-01/anat/upsample/{sub}_rim-LH.nii.gz'

    # =========================================================================
    # LN2_LAYERS
    command = f'LN2_LAYERS '
    command += f'-rim {rimFile} '
    command += f'-nr_layers 12 '
    # command += f'-curvature '
    command += f'-thickness '
    # command += f'-streamlines '
    command += '-equivol'

    subprocess.run(command, shell = True)

    # =========================================================================
    # Making control points file

    # Get dimensions of anatomy
    file = f'{subDir}/{sub}_seg_rim_trunc_polished_upsampled_midGM_equivol.nii.gz'
    nii = nb.load(file)
    affine = nii.affine
    header = nii.header
    cp = nii.get_fdata()

    cp[controlPoints[sub][0], controlPoints[sub][1], controlPoints[sub][2]] = 2

    cpFile = f'{subDir}/{sub}_seg_rim_trunc_polished_upsampled_midGM_equivol_cp.nii.gz'
    # Save control point
    ni_img = nb.Nifti1Image(cp.astype('int'), affine=affine, header=header)
    nb.save(ni_img, f'{cpFile}')


    # =========================================================================
    # LN2_Multilaterate
    cpFile = f'{subDir}/ses-01/anat/upsample/sub-09_rim-LH_midGM_equivol_cp.nii.gz'

    command = f'LN2_MULTILATERATE '
    command += f'-rim {rimFile} '
    command += f'-control_points {cpFile} '
    command += f'-radius 1'

    subprocess.run(command, shell = True)


    # =========================================================================
    # Flattening

    # base = f'{subDir}/{sub}_seg_rim_trunc_polished_upsampled'
    #
    # command = f'LN2_PATCH_FLATTEN '
    # command += f'-coord_uv {base}_UV_coordinates.nii.gz '
    # command += f'-coord_d {base}_metric_equivol.nii.gz '
    # command += f'-domain {base}_perimeter_chunk.nii.gz '
    # command += f'-bins_u 1000 -bins_v 1000 -bins_d 100 '
    # command += f'-values {base}_curvature_binned.nii.gz '
    # command += f'-voronoi'
    #
    # subprocess.run(command, shell = True)
