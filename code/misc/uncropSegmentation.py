import nibabel as nb
import numpy as np

BBOX = {'sub-05': {'RH': {'xlower': 435, 'xrange': 162, 'ylower': 55, 'yrange': 162, 'zlower': 95, 'zrange': 158},
                   'LH': {'xlower': 263, 'xrange': 162, 'ylower': 35, 'yrange': 162, 'zlower': 79, 'zrange': 158}},
        'sub-06': {'LH': {'xlower': 271, 'xrange': 162, 'ylower': 7, 'yrange': 162, 'zlower': 31, 'zrange': 159}},
        'sub-07': {'LH': {'xlower': 271, 'xrange': 166, 'ylower': 35, 'yrange': 158, 'zlower': 23, 'zrange': 166}},
        'sub-08': {'LH': {'xlower': 275, 'xrange': 162, 'ylower': 15, 'yrange': 162, 'zlower': 47, 'zrange': 158}},
        'sub-09': {'RH': {'xlower': 415, 'xrange': 162, 'ylower': 11, 'yrange': 162, 'zlower': 91, 'zrange': 158},
                   'LH': {'xlower': 303, 'xrange': 162, 'ylower': 0, 'yrange': 162, 'zlower': 59, 'zrange': 158}}
                   }

SUBS = ['sub-08']

for sub in SUBS:
    for hemi in ['LH']:
        tmpBox = BBOX[sub][hemi]

        anatFolder = f'/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/{sub}/ses-01/anat/upsample'
        gmFile = f'{anatFolder}/{sub}_rim-{hemi}_perimeter_chunk.nii'
        anatFile = f'{anatFolder}/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'

        base = gmFile.split('.')[0].split('/')[-1]

        anat = nb.load(anatFile)
        header = anat.header
        affine = anat.affine
        anatData = anat.get_fdata()

        gm = nb.load(gmFile)
        gmData = gm.get_fdata()
        gmData.shape

        new = np.zeros(anatData.shape)
        new[tmpBox['xlower']:tmpBox['xlower']+tmpBox['xrange'],
            tmpBox['ylower']:tmpBox['ylower']+tmpBox['yrange'],
            tmpBox['zlower']:tmpBox['zlower']+tmpBox['zrange']] = gmData

        img = nb.Nifti1Image(new, header=header,affine=affine)
        nb.save(img, f'{anatFolder}/{base}_uncrop.nii.gz')
