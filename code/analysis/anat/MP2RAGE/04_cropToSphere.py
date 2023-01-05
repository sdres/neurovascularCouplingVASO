
'''

Cropping upsampled UNI image to sphere.

'''
import os
import subprocess
import glob

SUBS = ['sub-05']

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

BBOX = {'sub-05': {'RH': {'xlower': 435, 'xrange': 162, 'ylower': 55, 'yrange': 162, 'zlower': 95, 'zrange': 158},
                   'LH': {'xlower': 263, 'xrange': 162, 'ylower': 35, 'yrange': 162, 'zlower': 79, 'zrange': 158}},
        'sub-06': {'LH':{'xlower': 271, 'xrange': 162, 'ylower': 7, 'yrange': 162, 'zlower': 31, 'zrange': 159}},
        'sub-07': {'LH':{'xlower': 271, 'xrange': 166, 'ylower': 35, 'yrange': 158, 'zlower': 23, 'zrange': 166}},
        'sub-08': {'LH':{'xlower': 275, 'xrange': 162, 'ylower': 15, 'yrange': 162, 'zlower': 47, 'zrange': 158}},
        'sub-09': {'RH':{'xlower': 415, 'xrange': 162, 'ylower': 11, 'yrange': 162, 'zlower': 91, 'zrange': 158},
                   'LH':{'xlower': 303, 'xrange': 162, 'ylower': 0, 'yrange': 162, 'zlower': 59, 'zrange': 158}}
        }

for sub in SUBS:

    # =========================================================================
    # Crop UNI to sphere

    # Find upsampled UNI images in first sessions
    image = glob.glob(f'{DATADIR}/{sub}/ses-01/anat/upsample/*uni*ups4X.nii.gz')[0]
    # image = glob.glob(f'/Users/sebastiandresbach/Desktop/forFaruk/sub-06_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_T2s_registered.nii.gz')[0]
    base = image.split('.')[0]

    # for hemi in ['RH','LH']:
    for hemi in ['LH']:
        tmpBBOX = BBOX[sub][hemi]


        command = 'fslroi '
        command += f'{image} '
        command += f'{base}_crop-toSphere{hemi}.nii.gz '
        command += f"{tmpBBOX['xlower']} {tmpBBOX['xrange']} {tmpBBOX['ylower']} {tmpBBOX['yrange']} {tmpBBOX['zlower']} {tmpBBOX['zrange']}"
        # break
        subprocess.run(command, shell = True)

        # =========================================================================
        # Give sphere the same dimensions

        # Find upsampled sphere image
        image = glob.glob(f'{DATADIR}/{sub}/ses-01/anat/upsample/{sub}_{hemi}_sphere_ups4X.nii.gz')[0]
        base = image.split('.')[0]

        # tmpBBOX = BBOX[sub]

        # base = os.path.basename(image).rsplit('.', 2)[0]

        command = 'fslroi '
        command += f'{image} '
        command += f'{base}_crop-toSphere{hemi}.nii.gz '
        command += f"{tmpBBOX['xlower']} {tmpBBOX['xrange']} {tmpBBOX['ylower']} {tmpBBOX['yrange']} {tmpBBOX['zlower']} {tmpBBOX['zrange']}"
        # break
        subprocess.run(command, shell = True)



        # Find upsampled sphere image
        image = glob.glob(f'{DATADIR}/{sub}/{sub}_ses-avg_task-stimulation_run-avg_part-mag_T1w_registered.nii')[0]
        base = image.split('.')[0]

        # tmpBBOX = BBOX[sub]

        # base = os.path.basename(image).rsplit('.', 2)[0]

        command = 'fslroi '
        command += f'{image} '
        command += f'{base}_crop-toSphere{hemi}.nii.gz '
        command += f"{tmpBBOX['xlower']} {tmpBBOX['xrange']} {tmpBBOX['ylower']} {tmpBBOX['yrange']} {tmpBBOX['zlower']} {tmpBBOX['zrange']}"
        # break
        subprocess.run(command, shell = True)
