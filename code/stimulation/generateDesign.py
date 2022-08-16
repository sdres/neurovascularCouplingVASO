"""Generate stimulation protocol."""

import numpy as np
import pandas as pd
import time

dateNow = time.strftime("%Y-%m-%d_%H.%M")

# get the path that this script is in and change dir to it
# _thisDir = os.path.dirname(os.path.abspath(__file__))  # get current path
# os.chdir(_thisDir)  # change directory to this path

folder = '/Users/sebastiandresbach/git/neurovascularCouplingVASO/code/stimulation'

# =============================================================================
# Chose timings for the stimuli
# =============================================================================

# Set pair TR (repetition time)
TR = 3
# Set number of jitters
nrJitters = 6

# We have 5 stimulus durations (in seconds)
stimDurs = [1, 2, 4, 12, 24]
# stimDurs =  [1, 2, 3, 4, 5]
nrStims = len(stimDurs)
stimDurArr = np.ones(nrJitters)

for i, stimDur in enumerate(stimDurs[1:]):
    for n in range(nrJitters):
        stimDurArr = np.append(stimDurArr, stimDur)

# Rest periods for these stimuli (in seconds)
restDurs = [12, 14, 16, 20, 24]
# restDurs = [2, 2, 2, 2, 2]
restDurArr = np.ones(nrJitters)*restDurs[0]

for i, restDur in enumerate(restDurs[1:]):
    for n in range(nrJitters):
        restDurArr = np.append(restDurArr, restDur)

# Steps between jitters
jitters = np.arange(0, nrJitters)
jitters = jitters * (TR/nrJitters)
jittersArr = jitters.copy()

for n in range(nrStims-1):
    jittersArr = np.append(jittersArr, jitters)

# Make a dataframe from all the possible conditions
conditions = pd.DataFrame({'stimDur': stimDurArr,
                           'restDur': restDurArr,
                           'jitter': jittersArr})

# Shuffle the dataframe
conditions = conditions.sample(frac=1).reset_index(drop=True)

# Save the dataframe
conditions.to_csv(
    f'{folder}/conditionTimings_TR-{TR}_jitters-{nrJitters}_{dateNow}.csv',
    index=False
    )

# Calculate run duration
stimDurTotal = np.sum(conditions['stimDur'])  # total stim duration
restDurTotal = np.sum(conditions['restDur'])  # total rest duration
jitterDurTotal = np.sum(conditions['jitter'])  # total rest duration

# Sum
expDurtotal = stimDurTotal + restDurTotal + jitterDurTotal + 15 + 50  # initial and end rest
print(f'Total experiment time: {expDurtotal} seconds')

TRs = expDurtotal/TR
