'''

Defining ROI masks. This is done in several steps:
- Select voxels from activation map based on threshold. Here, we chose half the
maximum z-score as cut-off
- Propagate active voxels across cortical depth using LayNii's UVD filter
- Remove floating bits that might occur in the process

'''

import numpy as np
import nibabel as nb
import os
import glob
import matplotlib.pyplot as plt
import subprocess
import os.path
from skimage.measure import label

# Set data path
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

SUBS = ['sub-05']

# =============================================================================
# Threshold and binarize activation
# =============================================================================

for sub in SUBS:
    mapFolder = f'{ROOT}/{sub}/statMaps'
    outFolder = f'{ROOT}/{sub}/rois'

    # Create folder if not exists
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
        print("Output directory is created")

    mapFile = f'{mapFolder}/{sub}_vaso_stimulation_registered_crop.nii.gz'

    outBase = os.path.basename(mapFile).split('.')[0]

    mapNii = nb.load(mapFile)
    mapData = mapNii.get_fdata()
    thr = np.max(mapData)/3
    mapData = mapData >= thr

    # data = label(mapData, connectivity=1)
    # labels, counts = np.unique(data, return_counts=True)
    # largestCluster = np.argmax(counts[1:])+1
    #
    # tmp = data == largestCluster

    img = nb.Nifti1Image(mapData, affine = mapNii.affine, header = mapNii.header)
    nb.save(img, f'{outFolder}/{outBase}_largestCluster_bin.nii.gz')


# =============================================================================
# Propagate ROI across cortical depth if voxel in GM
# =============================================================================

for sub in SUBS:

    anatDir = f'{ROOT}/{sub}/segmentation'
    # funcDir = f'{ROOT}/derivatives/{sub}/func'


    outFolder = f'{ROOT}/{sub}/rois'





    mapFile = f'{outFolder}/{sub}_vaso_stimulation_registered_crop_largestCluster_bin.nii.gz'
    baseName = mapFile.split('/')[-1].split('.')[0]

    command = 'LN2_UVD_FILTER '
    command += f'-values {mapFile} '
    command += f'-coord_uv {anatDir}/{sub}_uni_part-mag_avg_MP2RAGE_brain_ups4X_LH_sphere_crop_pveseg_corrected_polished_cleanBorders_polished_corrected_polished_UV_coordinates.nii.gz '
    command += f'-coord_d {anatDir}/sub-05_uni_part-mag_avg_MP2RAGE_brain_ups4X_LH_sphere_crop_pveseg_corrected_polished_cleanBorders_polished_corrected_polished_metric_equivol.nii.gz '
    command += f'-domain {anatDir}/sub-05_uni_part-mag_avg_MP2RAGE_brain_ups4X_LH_sphere_crop_pveseg_corrected_polished_cleanBorders_polished_corrected_polished_perimeter_chunk.nii.gz '
    command += f'-radius 0.45 '
    command += f'-height 2 '
    command += f'-max'

    subprocess.run(command,shell=True)
    print(f'Done with {sub}')
