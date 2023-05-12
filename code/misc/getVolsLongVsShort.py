"""Find the number of volumes we acquired for long vs short ITI sessions"""

import nibabel as nb
import glob

DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']

for sub in SUBS:
    print('')

    # =========================================================================
    # Look for sessions
    # Collect all runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))

    # Initiate list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1, 6):  # We had a maximum of 5 sessions
            if f'ses-0{i}' in run:
                sessions.append(f'ses-0{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))

    for ses in sessions:
        run = sorted(glob.glob(f'{DATADIR}/{sub}/{ses}/func/'
                               f'{sub}_{ses}_task-stimulation_run-avg_part-mag_vaso_intemp.nii.gz'))[0]

        nii = nb.load(run)
        header = nii.header
        vols = header['dim'][4]

        print(f'{sub} {ses} {vols}')
