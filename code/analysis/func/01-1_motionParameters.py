'''

Read and plot motion traces

'''

import ants
import os
import glob
import nibabel as nb
import numpy as np
import subprocess
from IPython.display import clear_output
import nipype.interfaces.fsl as fsl
import itertools
import pandas as pd


def my_ants_affine_to_distance(affine, unit):

    dx, dy, dz = affine[9:]

    rot_x = np.arcsin(affine[6])
    cos_rot_x = np.cos(rot_x)
    rot_y = np.arctan2(affine[7] / cos_rot_x, affine[8] / cos_rot_x)
    rot_z = np.arctan2(affine[3] / cos_rot_x, affine[0] / cos_rot_x)

    if unit == 'deg':
        deg = np.degrees
        R = np.array([deg(rot_x), deg(rot_y), deg(rot_z)])
    if unit == 'rad':
        R = np.array([rot_x, rot_y, rot_z])

    T = np.array([dx, dy, dz])

    return T, R


subs = ['sub-05']
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
ses = 'ses-01'



for sub in subs:

    funcDir = f'{ROOT}/derivatives/{sub}/ses-01/func'

    # look for individual runs (containing both nulled and notnulled images)
    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-*run-0*_part-mag_*.nii*'))


    motionTraces = []
    motionNames = []
    volumes = []
    modalityList = []
    runList = []

    # Set folder where motion traces were dumped
    motionDir = f'{funcDir}/motionParameters'

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0]
        print(f'Processing run {base}')



        modality = base.split('_')[-1]
        data = pd.read_csv(f'{motionDir}/{base}/{base}_{modality}_motionRegressors.csv')

        for col in data.columns:
            tmp = data[col].to_numpy()

            for i, val in enumerate(tmp, start = 1):
                motionTraces.append(val)
                motionNames.append(f'{col} {modality}')
                volumes.append(i)
                modalityList.append(modality)
                runList.append(base)

    data_dict = {
    'volume': volumes,
    'Motion': motionTraces,
    'name': motionNames,
    'modality': modalityList,
    'run': runList
        }

    pd_ses = pd.DataFrame(data=data_dict)
    pd_ses.to_csv(os.path.join(motionDir, f'{base}_motionSummary.csv'), index=False)





for sub in subs:

    funcDir = f'{ROOT}/derivatives/{sub}/{ses}/func'

    # look for individual runs (containing both nulled and notnulled images)
    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0]
        print(f'Processing run {base}')

        # Set folder where motion traces were dumped
        motionDir = f'{funcDir}/motionParameters/{base}'

        # Find modality of run
        modality = base.split('_')[-1]
        mats = sorted(glob.glob(f'{motionDir}/{base}_vol*'))

        Tr = []; Rt = []

        for i, mat in enumerate(mats):

            localtxp = ants.read_transform(mat)
            affine = localtxp.parameters

            T, R = my_ants_affine_to_distance(affine, 'rad')

            Tr.append(T)
            Rt.append(R)


        # // Save motion traces intra-run as .csv
        Tr = np.asarray(Tr)
        Rt = np.asarray(Rt)

        data_dict = {
        'Tx': Tr[:, 0],
        'Ty': Tr[:, 1],
        'Tz': Tr[:, 2],
        'Rx': Rt[:, 0],
        'Ry': Rt[:, 1],
        'Rz': Rt[:, 2]
        }

        pd_ses = pd.DataFrame(data=data_dict)
        pd_ses.to_csv(os.path.join(motionDir, f'{base}_{modality}_motionRegressors.csv'), index=False)



for sub in subs:

    funcDir = f'{ROOT}/derivatives/{sub}/{ses}/func'

    # look for individual runs (containing both nulled and notnulled images)
    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))


    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')
        # Set folder where motion traces were dumped
        motionDir = f'{funcDir}/motionParameters/{base}'

        sub_FD = []
        timepoints = []
        subjects=[]
        mods = []
        runList = []


        modality = base.split('_')[-1]

        data = pd.read_csv(os.path.join(motionDir, f'{base}_{modality}_motionRegressors.csv'))

        TX = data['Tx'].to_numpy()
        TY = data['Ty'].to_numpy()
        TZ = data['Tz'].to_numpy()
        RX = data['Rx'].to_numpy()
        RY = data['Ry'].to_numpy()
        RZ = data['Rz'].to_numpy()

        for n in range(len(TX)-4):
            FD_trial = abs(TX[n]-TX[n+1])+abs(TY[n]-TY[n+1])+abs(TZ[n]-TZ[n+1])+abs((50*RX[n])-(50*RX[n+1]))+abs((50*RY[n])-(50*RY[n+1]))+abs((50*RZ[n])-(50*RZ[n+1]))
            sub_FD.append(FD_trial)
            timepoints.append(n)
            subjects.append(sub)
            mods.append(modality)
            # runList.append(base)


    FDs = pd.DataFrame({'subject':subjects, 'volume':timepoints, 'FD':sub_FD, 'modality': mods})
    FDs.to_csv(os.path.join(motionDir, f'{base}_FDs.csv'), index=False)
