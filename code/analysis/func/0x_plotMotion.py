import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

root = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'
plt.style.use('dark_background')
v1Palette = {
    'bold': 'tab:orange',
    'cbv': 'tab:blue'}


motionPalette = ['Set1', 'Set2']

for sub in ['sub-01']:
    for ses in ['ses-01']:
        runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-*run-0*part-mag_cbv.nii.gz'))

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(f'Processing run {base}')

            outFolder = f'{root}/derivatives/{sub}/{ses}'

            fig, axes = plt.subplots(1, 2,sharex=True,figsize=(30,6))
            plt.suptitle(f'{base} Motion Summary', fontsize=24)

            width = 2

            for j, modality in enumerate(['bold', 'cbv']):
                rotTrans = []
                newName = []

                motionData = pd.read_csv(f'{outFolder}/motionParameters/{base}_{modality}_motionParameters.csv')
                for i, row in motionData.iterrows():
                    if 'R' in row['Motion_Name']:
                        rotTrans.append('rotation')
                        newName.append(row['Motion_Name'][1:])
                    if 'T' in row['Motion_Name']:
                        rotTrans.append('translation')
                        newName.append(row['Motion_Name'][1:])

                motionData['type'] = rotTrans
                motionData['Motion_Name'] = newName

                motionData_rot_nulled = motionData.loc[(motionData['type'].str.contains("rotation") == 1)].dropna()
                motionData_trans_nulled = motionData.loc[(motionData['type'].str.contains("translation") == 1)].dropna()

                sns.lineplot(ax=axes[0], x='Time/TR',y='Motion',data=motionData_trans_nulled, hue='Motion_Name', palette = motionPalette[j],linewidth = width,legend=False)

                axes[0].set_ylabel("Translation [mm]", fontsize=24)
                axes[0].set_xlabel("Volume", fontsize=24)

                sns.lineplot(ax=axes[1], x='Time/TR',y='Motion',data=motionData_rot_nulled,hue='Motion_Name', palette = motionPalette[j],linewidth = width)

                axes[1].set_xlabel("Volume", fontsize=24)
                axes[1].set_ylabel("Rotation [radians]", fontsize=24)
                axes[1].legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
                axes[0].tick_params(axis='both', labelsize=20)
                axes[1].tick_params(axis='both', labelsize=20)

            # mylabels = ['X VASO','Y VASO','Z VASO','X BOLD','Y BOLD','Z BOLD']

            plt.savefig(f'../../results/motionParameters/{base}_motion.jpg', bbox_inches = 'tight', pad_inches = 0)
            plt.show()


# plot framewise displacements

for sub in ['sub-02']:
    for ses in ['ses-01']:
        runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-*run-0*part-mag_cbv.nii.gz'))

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(f'Processing run {base}')

            outFolder = f'{root}/derivatives/{sub}/{ses}'

            fig = plt.figure(figsize=(20,5))
            plt.title(f"{base}", fontsize=24, pad=20)

            sub_FD = []
            timepoints = []
            subjects=[]
            mods = []
            runList = []

            for j, modality in enumerate(['bold', 'cbv']):
                rotTrans = []
                newName = []

                motionData = pd.read_csv(f'{outFolder}/motionParameters/{base}_{modality}_motionParameters.csv')
                for i, row in motionData.iterrows():
                    if 'R' in row['Motion_Name']:
                        rotTrans.append('rotation')
                        newName.append(row['Motion_Name'][1:])
                    if 'T' in row['Motion_Name']:
                        rotTrans.append('translation')
                        newName.append(row['Motion_Name'][1:])

                motionData['type'] = rotTrans
                motionData['Motion_Name'] = newName


                TX = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("X") == 1)&(motionData['type']=='translation')].tolist()
                TY = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("Y") == 1)&(motionData['type']=='translation')].tolist()
                TZ = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("Z") == 1)&(motionData['type']=='translation')].tolist()

                RX = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("X") == 1)&(motionData['type']=='rotation')].tolist()
                RY = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("Y") == 1)&(motionData['type']=='rotation')].tolist()
                RZ = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("Z") == 1)&(motionData['type']=='rotation')].tolist()

                for n in range(len(TX)-4):
                    FD_trial = abs(TX[n]-TX[n+1])+abs(TY[n]-TY[n+1])+abs(TZ[n]-TZ[n+1])+abs((50*RX[n])-(50*RX[n+1]))+abs((50*RY[n])-(50*RY[n+1]))+abs((50*RZ[n])-(50*RZ[n+1]))
                    sub_FD.append(FD_trial)
                    timepoints.append(n)
                    subjects.append(sub)
                    mods.append(modality)
                    # runList.append(base)


            FDs = pd.DataFrame({'subject':subjects, 'volume':timepoints, 'FD':sub_FD, 'modality': mods})
            FDs.to_csv(f'{root}/derivatives/{sub}/{ses}/motionParameters/{base}_FDs.csv', index=False)

            sns.lineplot(data=FDs, x='volume', y='FD',hue='modality',linewidth = width, palette=v1Palette)

            if np.max(FDs['FD']) < 0.9:
                plt.ylim(0,1)


            plt.axhline(0.9, color='gray', linestyle='--')
            plt.ylabel('FD [mm]', fontsize=24)
            plt.xlabel('Volume', fontsize=24)
            plt.legend(fontsize=20)


            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.savefig(f'../../results/motionParameters/{base}_FDs.png', bbox_inches = 'tight', pad_inches = 0)
            plt.show()
