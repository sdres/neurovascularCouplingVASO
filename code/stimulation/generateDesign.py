import numpy as np
import random
import pandas as pd
import time
import os

dateNow = time.strftime("%Y-%m-%d_%H.%M")

# get the path that this script is in and change dir to it
_thisDir = os.path.dirname(os.path.abspath(__file__))  # get current path
os.chdir(_thisDir)  # change directory to this path


#########################################
##### Chose timings for the stimuli #####
#########################################

# we have 5 stimulus durations
stimDurs =  [2, 4, 12, 24] # stimulus durations in seconds
stimDurArr = np.ones(5)

for i, stimDur in enumerate(stimDurs):
    for n in range(5):
        stimDurArr = np.append(stimDurArr, stimDur)

# The rest periods for these stimuli are
restDurs = [12, 16, 20, 24]  # rest durations in seconds
restDurArr = np.ones(5)*10

for i, restDur in enumerate(restDurs):
    for n in range(5):
        restDurArr = np.append(restDurArr, restDur)

# We want to have 5 sub TR jitters
# These have to be determined wrt the TR
# Here we assume a TR of 3.3 s

TR = 3.3

# therefore the steps between jitters are
jitters = np.arange(0,5)
jitters = jitters * (TR/5)
jittersArr = jitters.copy()
for n in range(4):
    jittersArr = np.append(jittersArr, jitters)

# lets make a daraframe from all the possible conditions
conditions = pd.DataFrame({'stimDur':stimDurArr, 'restDur': restDurArr, 'jitter': jittersArr})
conditions = conditions.sample(frac=1).reset_index(drop=True)

conditions.to_csv(f'/Users/sebastiandresbach/git/neurovascularCouplingVASO/code/stimulation/conditionTimings_{dateNow}.csv', index=False)

stimDurTotal = np.sum(conditions['stimDur'])
restDurTotal = np.sum(conditions['restDur'])

expDurtotal = stimDurTotal + restDurTotal + 30 + 5 # 30s initial and 5s end rest


#######################################################
##### Chose timings for the attenton-task targets #####
#######################################################


min = 10 # min duration between targets
max = 20 # max duration between targets
nrTargets = int(expDurtotal/6) # present one target per minute

targetTimes = s = np.random.uniform(min,max,nrTargets)


np.savetxt("/Users/sebastiandresbach/git/neurovascularCouplingVASO/code/stimulation/targetTimes.csv", targetTimes, delimiter= " ", fmt = '%1.3f')
