"""

Cropping functional images to decrease number of voxels (and therefore
processing demands)

"""

from nipype.interfaces.fsl import ExtractROI
import glob
import os

# Set data path
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

# Set subjects to work on
subs = ['sub-06']

# Loop over participants
for sub in subs:
    for modality in ['vaso', 'bold']:
        runs = sorted(glob.glob(
            f'{ROOT}/derivatives/{sub}/*/func/{sub}_ses-*_task-stimulation_run-avg_part-mag_{modality}_intemp.nii*'))

        # Loop over runs
        for run in runs:

            # Get basename of run
            base = os.path.basename(run).rsplit('.', 2)[0]
            print(f'Processing run {base}')

            for i in range(1, 99):
                if f'ses-0{i}' in run:
                    ses = f'ses-0{i}'

            # Define output folder
            outFolder = f'{ROOT}/derivatives/{sub}/{ses}/func'

            inFile = f'{run}'
            outFile = f'{outFolder}/{base}_trunc.nii.gz'

            # Prepare command
            fslroi = ExtractROI(in_file=inFile,
                                roi_file=outFile,
                                x_min=0, x_size=148,
                                y_min=0, y_size=74,
                                z_min=0, z_size=16
                                )
            # Run command
            out = fslroi.run()

    #
    # runs = sorted(glob.glob(
    #     f'{ROOT}/derivatives/{sub}/*/func/{sub}_ses-*_task-stimulation_run-avg_part-mag_.nii*'))
    # # Do the same for t1w images
    # inFile = f'{outFolder}/{sub}_{ses}_T1w.nii'
    # outFile = f'{outFolder}/{sub}_{ses}_T1w_trunc.nii.gz'
    #
    # fslroi = ExtractROI(in_file=inFile,
    #                     roi_file=outFile,
    #                     x_min=boundariesDict[sub][0], x_size=boundariesDict[sub][1],
    #                     y_min=boundariesDict[sub][2], y_size=boundariesDict[sub][3],
    #                     z_min=boundariesDict[sub][4], z_size=boundariesDict[sub][5]
    #                     )
    #
    # out = fslroi.run()
