import ants
import os
import glob
from nipype.interfaces import afni
import nibabel as nb
import numpy as np
import subprocess
from IPython.display import clear_output
import nipype.interfaces.fsl as fsl
import itertools
import pandas as pd

root = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
antsPath = '/Users/sebastiandresbach/ANTs/install/bin'
afniPath = '/Users/sebastiandresbach/abin'


for sub in ['sub-02']:
    os.system(f'mkdir {root}/derivatives/{sub}')
    # for ses in ['ses-01', 'ses-02']:
    for ses in ['ses-01']:
        os.system(f'mkdir {root}/derivatives/{sub}/{ses}')

        # look for individual runs
        runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-0*_part-mag_*.nii.gz'))
        # make folder to dump motion traces
        outFolder = f'{root}/derivatives/{sub}/{ses}'
        os.system(f'mkdir {outFolder}/motionParameters')

        for j, run in enumerate(runs,start=1):
            base = os.path.basename(run).rsplit('.', 2)[0]
            print(f'Processing run {base}')

            modality = base.split('_')[-1]

            nii = nb.load(run)
            # get header and affine
            header = nii.header
            affine = nii.affine
            # Load data as array
            data = nii.get_fdata()

            # make folder to dump motion traces for the run
            os.system(f'mkdir {outFolder}/motionParameters/{base}')

            # initiate lists for motion traces
            lstsubMot = [] # List for the motion value
            lstsubMot_Nme = [] # List for the motion name (e.g. translation in x direction)
            lstTR_sub = [] # List for name of subject. technically not needed because we are only doing it run by run
            modalityList = [] # List for nulled/notnulled

            # make moma
            print('Generating mask')
            subprocess.run(f'{afniPath}/3dAutomask -prefix {outFolder}/{base}_moma.nii -peels 3 -dilate 2 {root}/{sub}/{ses}/func/{base}.nii.gz',shell=True)

            # make reference image
            reference = np.mean(data[:,:,:,4:6],axis=-1)
            # and save it
            img = nb.Nifti1Image(reference, header=header, affine=affine)
            nb.save(img, f'{outFolder}/{base}_reference.nii')

            # separate into individual volumes
            for i in range(data.shape[-1]):
                # overwrite volumes 0,1,2 with volumes 3,4,5
                if i <= 2:
                    vol = data[:,:,:,i+3]
                else:
                    vol = data[:,:,:,i]
                # Save individual volumes
                img = nb.Nifti1Image(vol, header=header, affine=affine)
                nb.save(img, f'{outFolder}/{base}_vol{i:03d}.nii')
            # define mask and reference images in 'antspy-style'
            fixed = ants.image_read(f'{outFolder}/{base}_reference.nii')
            mask = ants.image_read(f'{outFolder}/{base}_moma.nii')

            # loop over volumes to do the correction
            for i in range(data.shape[-1]):
                moving = ants.image_read(f'{outFolder}/{base}_vol{i:03d}.nii')
                mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform = 'Rigid', mask=mask)
                # save transformation matrix for later
                os.system(f"cp {mytx['fwdtransforms'][0]} {outFolder}/motionParameters/{base}/{base}_vol{i:03d}.mat")
                # convert transformattion matrix into FSL format
                os.system(f'{antsPath}/ConvertTransformFile 3 {outFolder}/motionParameters/{base}/{base}_vol{i:03d}.mat {outFolder}/motionParameters/{base}/{base}_vol{i:03d}_af.mat --convertToAffineType')
                os.system(f'/usr/local/bin/c3d_affine_tool -ref {outFolder}/{base}_reference.nii -src {outFolder}/{base}_vol{i:03d}.nii -itk {outFolder}/motionParameters/{base}/{base}_vol{i:03d}_af.mat -ras2fsl -o {outFolder}/motionParameters/{base}/{base}_vol{i:03d}_FSL.mat -info-full')
                # read parameters
                tmp = fsl.AvScale(all_param=True,mat_file=f'{outFolder}/motionParameters/{base}/{base}_vol{i:03d}_FSL.mat');
                tmpReadout = tmp.run();

                # Get the rotations (in rads) and translations (in mm) per volume
                aryTmpMot = list(itertools.chain.from_iterable([tmpReadout.outputs.translations, tmpReadout.outputs.rot_angles]));

                # Save the rotation and translations in lists
                lstsubMot.append(aryTmpMot)
                lstTR_sub.append([int(i)+1 for k in range(6)])
                lstsubMot_Nme.append([f'TX {modality}',f'TY {modality}',f'TZ {modality}',f'RX {modality}',f'RY {modality}',f'RZ {modality}'])
                modalityList.append([modality for k in range(6)])

                clear_output(wait=True)
                # apply transformation
                mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving,transformlist=mytx['fwdtransforms'], interpolator='bSpline')
                # save warped image
                ants.image_write(mywarpedimage, f'{outFolder}/{base}_vol{i:03d}_warped.nii')

            # assemble images
            newData = np.zeros(data.shape)
            for i in range(data.shape[-1]):
                vol = nb.load(f'{outFolder}/{base}_vol{i:03d}_warped.nii').get_fdata()
                newData[:,:,:,i] = vol
            img = nb.Nifti1Image(newData, header=header, affine=affine)
            nb.save(img, f'{outFolder}/{base}_moco.nii')
            # remove volumes
            os.system(f'rm {outFolder}/{base}_vol*.nii')


            # Make appropriate arrays from lists
            aryCurr = np.array(lstsubMot)
            aryCurr_Ses =  aryCurr.reshape((aryCurr.size,-1))

            aryCurr_TR = np.array(lstTR_sub)
            aryCurr_TR_Ses = aryCurr_TR.reshape((aryCurr_TR.size,-1))

            aryCurr_Nme = np.array(lstsubMot_Nme)
            aryCurr_Nme_Ses = aryCurr_Nme.reshape((aryCurr_Nme.size,-1))

            aryIdx = np.arange(1,len(aryCurr_Nme_Ses)+1)

            aryCurr_mod = np.array(modalityList)
            aryCurr_mod = aryCurr_mod.reshape((aryCurr_mod.size,-1))

            data_dict = {
                'Time/TR': aryCurr_TR_Ses[:,0],
                'Motion_Name': aryCurr_Nme_Ses[:,0],
                'Motion': aryCurr_Ses[:,0],
                'idx':aryIdx,
                'modality': aryCurr_mod[:,0]}

            # Save motion parameters as csv
            pd_ses = pd.DataFrame(data=data_dict)
            pd_ses.to_csv(f'{outFolder}/motionParameters/{base}_motionParameters.csv', index=False)


# get T1w image in EPI space
for sub in ['sub-02']:
    # for ses in ['ses-01', 'ses-02']:
    for ses in ['ses-01']:
        outFolder = f'{root}/derivatives/{sub}/{ses}'
        # look for individual runs
        runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-0*_part-mag_*.nii.gz'))

        nrRuns = int(len(runs)/2)

        for runNr in range(1, nrRuns+1):
        # for runNr in range(1, 2):

            modalities = glob.glob(f'{outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_*_moco.nii')
            # combining cbv and bold weighted images
            os.system(f'{afniPath}/3dTcat -prefix {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_combined.nii  {modalities[0]} {modalities[1]} -overwrite')
            # Calculating T1w image in EPI space for each run
            os.system(f'{afniPath}/3dTstat -cvarinv -overwrite -prefix {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_T1w.nii {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_combined.nii')
            # Running biasfieldcorrection
            os.system(f'{antsPath}/N4BiasFieldCorrection -d 3 -i {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_T1w.nii -o {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_T1w_N4Corrected.nii')


# register all T1w images to first run of ses-01
for sub in ['sub-02']:
    # for ses in ['ses-01', 'ses-02']:
    for ses in ['ses-01']:
        outFolder = f'{root}/derivatives/{sub}/{ses}'

        # look for individual runs
        runs = sorted(
            glob.glob(
            f'{root}/{sub}/{ses}/func' # folder
            + f'/{sub}_{ses}_task-stimulation_run-0*_part-mag_*.nii.gz' # files
            )
            )

        nrRuns = int(len(runs)/2)

        refImage = (
            f'{root}/derivatives/{sub}/ses-01/'
            + f'{sub}_ses-01_task-stimulation_run-01_part-mag_T1w_N4Corrected.nii'
            )

        refBase = os.path.basename(refImage).rsplit('.', 2)[0]

        fixed = ants.image_read(refImage)

        mask = ants.image_read(
            f'{root}/derivatives/{sub}/ses-01/'
            + f'{sub}_ses-01_task-stimulation_run-01_part-mag_cbv_moma.nii'
            )

        if ses == 'ses-01':
            firstRun = 2
        if ses == 'ses-02':
            firstRun = 1

        for runNr in range(firstRun, nrRuns+1):

            moving = ants.image_read(
                f'{outFolder}'
                + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_T1w_N4Corrected.nii'
                )

            mytx = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform='Rigid',
                mask=mask
                )

            # perform transofrmation
            mywarpedimage = ants.apply_transforms(
                fixed=fixed,
                moving=moving,
                transformlist=mytx['fwdtransforms'],
                interpolator='bSpline'
                )

            #save image
            ants.image_write(
                mywarpedimage,
                f'{outFolder}'
                + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_T1w_N4Corrected_registered-{refBase}.nii')

            # save transform for future
            transform1 = (
                f'{outFolder}'
                + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_T1w_N4Corrected_registered-{refBase}.mat'
                )

            os.system(f"cp {mytx['fwdtransforms'][0]} {transform1}")

############################################################################
############# Here, the coregistration of multiple runs starts #############
############################################################################

for sub in ['sub-02']:
    # for ses in ['ses-01', 'ses-02']:
    for ses in ['ses-01']:
        outFolder = f'{root}/derivatives/{sub}/{ses}'

        # look for individual runs
        runs = sorted(glob.glob(
            f'{root}/{sub}/{ses}/func'
            + f'/{sub}_{ses}_task-stimulation_run-0*_part-mag_*.nii.gz'
            )
            )


        nrRuns = int(len(runs)/2)

        refImage = f'{root}/derivatives/{sub}/ses-01/{sub}_ses-01_task-stimulation_run-01_part-mag_T1w_N4Corrected.nii'
        refHeader = nb.load(refImage).header
        refAffine = nb.load(refImage).affine

        refBase = os.path.basename(refImage).rsplit('.', 2)[0]

        fixed = ants.image_read(refImage)
        mask = ants.image_read(
            f'{root}/derivatives/{sub}/ses-01'
            + f'/{sub}_ses-01_task-stimulation_run-01_part-mag_cbv_moma.nii'
            )

        if ses == 'ses-01':
            firstRun = 2
        if ses == 'ses-02':
            firstRun = 1

        for runNr in range(firstRun, nrRuns+1):

            for modality in ['cbv', 'bold']:

                run = f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}.nii.gz'
                transform1 = f'{outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_T1w_N4Corrected_registered-{sub}_ses-01_task-stimulation_run-01_part-mag_T1w_N4Corrected.mat'

                nii = nb.load(run)
                # get header and affine
                header = nii.header
                affine = nii.affine
                # Load data as array
                data = nii.get_fdata()

                # separate into individual volumes
                for i in range(data.shape[-1]):
                    # overwrite volumes 0,1,2 with volumes 3,4,5
                    if i <= 2:
                        vol = data[:,:,:,i+3]
                    else:
                        vol = data[:,:,:,i]

                    # Save individual volumes
                    img = nb.Nifti1Image(vol, header=header, affine=affine)
                    nb.save(img,
                    f'{outFolder}'
                    + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_vol{i:03d}.nii')

                # loop over volumes to do the correction
                for i in range(data.shape[-1]):
                    moving = ants.image_read(
                        f'{outFolder}'
                        + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_vol{i:03d}.nii')

                    # get within run transoformation matrix
                    transform2 = f'{outFolder}/motionParameters/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_vol{i:03d}.mat'

                    # apply transofrmation matrices
                    mywarpedimage = ants.apply_transforms(
                        fixed=fixed,
                        moving=moving,
                        transformlist = [transform2, transform1],
                        interpolator = 'bSpline'
                        )

                    # save warped image
                    ants.image_write(mywarpedimage,
                        f'{outFolder}'
                        + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_vol{i:03d}_warped.nii'
                        )

                # assemble images
                newData = np.zeros(data.shape)
                for i in range(data.shape[-1]):
                    vol = nb.load(
                    f'{outFolder}'
                    + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_vol{i:03d}_warped.nii'
                    ).get_fdata()

                    newData[:,:,:,i] = vol

                img = nb.Nifti1Image(newData,
                    header=refHeader,
                    affine=refAffine
                    )

                nb.save(
                    img,
                    f'{outFolder}'
                    + f'/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_moco_registered.nii'
                    )

                # remove indvididual volumes
                os.system(f'rm {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_{modality}_vol*.nii')



# get registered T1w image in EPI space
for sub in ['sub-02']:
    # for ses in ['ses-01', 'ses-02']:
    for ses in ['ses-01']:

        # look for individual runs
        runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-stimulation_run-0*_part-mag_*.nii.gz'))

        nrRuns = int(len(runs)/2)

        if ses == 'ses-01':
            firstRun = 2
        if ses == 'ses-02':
            firstRun = 1

        for runNr in range(firstRun, nrRuns+1):

            modalities = glob.glob(f'{root}/derivatives/{sub}/{ses}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_*_moco_registered.nii')

            os.system(f'{afniPath}/3dTcat -prefix {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_registered_combined.nii  {modalities[0]} {modalities[1]} -overwrite')
            # Calculating T1w image in EPI space for each run
            os.system(f'{afniPath}/3dTstat -cvarinv -overwrite -prefix {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_registered_T1w.nii {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_registered_combined.nii')
            # Running biasfieldcorrection
            os.system(f'{antsPath}/N4BiasFieldCorrection -d 3 -i {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_registered_T1w.nii -o {outFolder}/{sub}_{ses}_task-stimulation_run-0{runNr}_part-mag_registered_T1w_N4Corrected.nii')
