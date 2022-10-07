'''

Creating stimulation times from log files and saving in bids format

'''

import numpy as np
import glob
import pandas as pd
import os
import re

import seaborn as sns
import matplotlib.pyplot as plt

# define root dir
root = '/Users/sebastiandresbach/git/neurovascularCouplingVASO'
# define subjects to work on
subs = ['sub-01']


for sub in subs:
    # get all runs of all sessions
    logFiles = sorted(glob.glob(f'{root}/code/stimulation/{sub}/ses-*/{sub}_ses-*_run-0*_neurovascularCoupling.log'))

    for logFile in logFiles:
        # get basename of current run
        base = os.path.basename(logFile).rsplit('.', 2)[0]
        # get logfile
        data = pd.read_csv(f'{logFile}'
                                ,usecols=[0])

        # Because the column definition will get hickups if empty colums are
        # present, we find line with first trigger to then load the file anew,
        # starting with that line
        for index, row in data.iterrows():
            if re.search('StartOfRun', str(row)):
                firstVolRow = index+1
                break
        # define column names
        ColNames = ['startTime', 'type', 'event']

        # load logfile again, starting with first trigger
        data = pd.read_csv(f'{logFile}', sep = '\t', skiprows=firstVolRow, names = ColNames)

        # initiate lists
        stimStart = []
        stimStop = []

        # loop over lines and fine stimulation start and stop times
        for index, row in data.iterrows():
            if not data['event'][index] != data['event'][index]:
                if re.search('stimulation started', data['event'][index]):
                    stimStart.append(data['startTime'][index])
                if re.search('stimulation stopped', data['event'][index]):
                    stimStop.append(data['startTime'][index])

        # convert lists to arrays and compute stimulation durations
        durs = np.asarray(stimStop) - np.asarray(stimStart)

        # make dataframe and save as text file
        design = pd.DataFrame({'startTime': stimStart, 'duration': durs, 'mod' : np.ones(len(durs))})
        np.savetxt(f'{root}/code/stimulation/{sub}/ses-01/{base}.txt', design.values, fmt='%1.2f')


#
#
# # We might also want to know the distribution of inter trial intervals in order
# # to check whether they are evenly distributed
#
# meanDur = np.mean(durs)
# ITIs = []
# for i, trial in enumerate(stimStart[:-1]):
#     tmp = (stimStart[i+1]-trial)-meanDur
#     ITIs.append(tmp)
#
# data = pd.DataFrame({'ITIs':ITIs})
#
# fig, ax = plt.subplots()
# sns.histplot(data=data, x="ITIs", binwidth=0.5)
# # sns.kdeplot(data=data, x="ITIs", bw_adjust=.25)
# plt.title(f'Inter-Trial Intervals',fontsize=24)
# plt.ylabel('# Trials',fontsize=24)
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14)
# plt.xlabel('ITI (seconds)',fontsize=20)
# plt.savefig(f'../results/ITIsCount.png',bbox_inches='tight')
# plt.show()
