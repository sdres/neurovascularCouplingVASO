"""Generate stimulation protocol."""

import numpy as np
import pandas as pd
import time

# Output folder
FOLDER = '/Users/sebastiandresbach/git/neurovascularCouplingVASO/code/stimulation'

# Pair TR (repetition time)
TR = 3
# Number of jitters
NR_JITTERS = 6

# Stimulus durations (in seconds)
STIM_DURS = [1, 2, 4, 12, 24]
# STIM_DURS =  [1, 2, 3, 4, 5]

# Rest durations (in seconds)
REST_DURS = [12, 14, 16, 20, 24]
# REST_DURS = [2, 2, 2, 2, 2]

# =============================================================================
dateNow = time.strftime("%Y-%m-%d_%H.%M")

nrStims = len(STIM_DURS)
stimDurArr = np.ones(NR_JITTERS)

for i, stimDur in enumerate(STIM_DURS[1:]):
    for n in range(NR_JITTERS):
        stimDurArr = np.append(stimDurArr, stimDur)

restDurArr = np.ones(NR_JITTERS)*REST_DURS[0]

for i, restDur in enumerate(REST_DURS[1:]):
    for n in range(NR_JITTERS):
        restDurArr = np.append(restDurArr, restDur)

# Steps between jitters
jitters = np.arange(0, NR_JITTERS)
jitters = jitters * (TR/NR_JITTERS)
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
    f'{FOLDER}/conditionTimings_TR-{TR}_jitters-{NR_JITTERS}_{dateNow}.csv',
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
