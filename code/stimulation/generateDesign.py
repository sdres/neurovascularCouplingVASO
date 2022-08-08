import numpy as np
import random
import pandas as pd
import time
import os

dateNow = time.strftime("%Y-%m-%d_%H.%M")

# get the path that this script is in and change dir to it
# _thisDir = os.path.dirname(os.path.abspath(__file__))  # get current path
# os.chdir(_thisDir)  # change directory to this path

folder = '/Users/sebastiandresbach/git/neurovascularCouplingVASO/code/stimulation'

#########################################
##### Chose timings for the stimuli #####
#########################################

# set pair TR
TR = 2
#set nr of jitters
nrJitters = 2

# we have 5 stimulus durations
# stimDurs =  [1, 2, 4, 12, 24] # stimulus durations in seconds
stimDurs =  [1, 2, 3, 4, 5] # stimulus durations in seconds
nrStims = len(stimDurs)
stimDurArr = np.ones(nrJitters)

for i, stimDur in enumerate(stimDurs[1:]):
    for n in range(nrJitters):
        stimDurArr = np.append(stimDurArr, stimDur)

# The rest periods for these stimuli are
# restDurs = [12, 14, 16, 20, 24]  # rest durations in seconds
restDurs = [2, 2, 2, 2, 2]  # rest durations in seconds
restDurArr = np.ones(nrJitters)*restDurs[0]

for i, restDur in enumerate(restDurs[1:]):
    for n in range(nrJitters):
        restDurArr = np.append(restDurArr, restDur)


# the steps between jitters are
jitters = np.arange(0,nrJitters)
jitters = jitters * (TR/nrJitters)
jittersArr = jitters.copy()

for n in range(nrStims-1):
    jittersArr = np.append(jittersArr, jitters)

# make a daraframe from all the possible conditions
conditions = pd.DataFrame(
    {
    'stimDur':stimDurArr,
    'restDur': restDurArr,
    'jitter': jittersArr
    }
    )

# shuffle the dataframe
conditions = conditions.sample(frac=1).reset_index(drop=True)

# save the dataframe
conditions.to_csv(
    f'{folder}/conditionTimings_TR-{TR}_jitters-{nrJitters}_{dateNow}_debug.csv',
    index=False
    )

# caculate run duration
stimDurTotal = np.sum(conditions['stimDur']) # total stim duration
restDurTotal = np.sum(conditions['restDur']) # total rest duration
jitterDurTotal = np.sum(conditions['jitter']) # total rest duration


# sum
expDurtotal = stimDurTotal + restDurTotal + jitterDurTotal + 5 + 10 # initial and end rest
print(f'Total experiment time: {expDurtotal} seconds')
