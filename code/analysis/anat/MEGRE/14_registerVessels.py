"""Register each run to one reference run."""

import os
import subprocess

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'
SUBS = ['sub-06']
BBOX = {'sub-05': {'RH': {'xlower': 435, 'xrange': 162, 'ylower': 55, 'yrange': 162, 'zlower': 95, 'zrange': 158},
                   'LH': {'xlower': 263, 'xrange': 162, 'ylower': 35, 'yrange': 162, 'zlower': 79, 'zrange': 158}},
        'sub-06': {'LH': {'xlower': 271, 'xrange': 162, 'ylower': 7, 'yrange': 162, 'zlower': 31, 'zrange': 159}},
        'sub-07': {'LH': {'xlower': 271, 'xrange': 166, 'ylower': 35, 'yrange': 158, 'zlower': 23, 'zrange': 166}},
        'sub-08': {'LH': {'xlower': 275, 'xrange': 162, 'ylower': 15, 'yrange': 162, 'zlower': 47, 'zrange': 158}},
        'sub-09': {'RH': {'xlower': 415, 'xrange': 162, 'ylower': 11, 'yrange': 162, 'zlower': 91, 'zrange': 158},
                   'LH': {'xlower': 303, 'xrange': 162, 'ylower': 0, 'yrange': 162, 'zlower': 59, 'zrange': 158}}
        }

for sub in SUBS:
    # Find MEGRE session
    for sesNr in range(1, 6):
        if os.path.exists(f"{DATADIR}/{sub}/ses-0{sesNr}/anat/megre/11_T2star/"
                          f"{sub}_ses-T2s_part-mag_MEGRE_crop_ups2X_prepped_avg_composite_decayfixed_S0.nii.gz"):
            megreSes = f'ses-0{sesNr}'

    # =============================================================================

    outDir = f"{DATADIR}/{sub}/{megreSes}/anat/megre/12_vessels"

    in_moving = f"{outDir}/{sub}_vessels.nii.gz"

    fixed = f'{DATADIR}/{sub}/ses-01/anat/upsample/' \
            f'{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'

    # Use ITK-SNAP manually to find the best registration
    initial = f"{DATADIR}/{sub}/{megreSes}/anat/megre/11_T2star/initial_matrix.txt"

    # =============================================================================
    print("Step_07: Apply registration from T2s to T1w space")

    # Prepare output
    basename, ext = in_moving.split(os.extsep, 1)
    basename = os.path.basename(basename)
    print(basename)
    out_moving = os.path.join(outDir, "{}_reg.nii.gz".format(basename))

    command2 = "greedy "
    command2 += "-d 3 "
    command2 += "-rf {} ".format(fixed)  # reference
    command2 += "-ri NN "
    command2 += "-rm {} {} ".format(in_moving, out_moving)  # moving resliced
    command2 += "-r {},-1 ".format(initial)
    print("{}\n".format(command2))

    # Execute command
    subprocess.run(command2, shell=True)

    # crop vessels to sphere
    base = out_moving.split('.')[0]

    tmpBBOX = BBOX[sub]['LH']

    command = 'fslroi '
    command += f'{out_moving} '
    command += f'{base}_crop-toSphereLH.nii.gz '
    command += f"{tmpBBOX['xlower']} " \
               f"{tmpBBOX['xrange']} " \
               f"{tmpBBOX['ylower']} " \
               f"{tmpBBOX['yrange']} " \
               f"{tmpBBOX['zlower']} " \
               f"{tmpBBOX['zrange']}"

    # break
    subprocess.run(command, shell=True)

print('\n\nFinished.')
