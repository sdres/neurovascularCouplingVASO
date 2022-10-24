'''

Removing non-brain tissue from MP2RAGE images to reduce data load.

'''

import os
import glob
import subprocess

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

SUBS = ['sub-05']

BBOX = {'sub-05': {'LH': {'xlower': 263, 'xrange': 162, 'ylower': 35, 'yrange': 162, 'zlower': 79, 'zrange': 158}}}


for sub in SUBS:

    # =========================================================================
    # Cropping
    roiDir = f'{DATADIR}/derivatives/{sub}/segmentation'


    # for hemi in ['LH', 'RH']:
    for hemi in ['LH']:

        images = sorted(glob.glob(f'{roiDir}/{sub}_uni_part-mag_avg_MP2RAGE_brain_ups4X_{hemi}_sphere.nii.gz'))

        tmpBBOX = BBOX[sub][hemi]

        for image in images[:1]:
            base = os.path.basename(image).rsplit('.', 2)[0]

            command = 'fslroi '
            command += f'{image} '
            command += f'{roiDir}/{base}_crop.nii.gz '
            command += f"{tmpBBOX['xlower']} {tmpBBOX['xrange']} {tmpBBOX['ylower']} {tmpBBOX['yrange']} {tmpBBOX['zlower']} {tmpBBOX['zrange']}"
            # break
            subprocess.run(command, shell = True)
