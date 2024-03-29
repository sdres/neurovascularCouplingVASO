"""Generate stimulation protocol."""

import numpy as np
import pandas as pd
import time

# Output folder
FOLDER = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation'

# Pair TR (repetition time)
TR = 3.141
# Number of jitters
NR_JITTERS = 4

# Stimulus durations (in seconds)
STIM_DURS = [1, 2, 4, 12, 24]
# STIM_DURS =  [1, 2, 3, 4, 5]  # Short stimulus durations for debugging

# Rest durations (in seconds)
REST_DURS = {'short': [12, 14, 16, 20, 24],
             'long': [20, 22, 24, 30, 40]}
# REST_DURS = [20, 22, 24, 30, 40]  # Long ITIs for baseline recovering

sub = 'sub-10'
ses = 'ses-01'
ITI = 'long'

# =============================================================================
dateNow = time.strftime("%Y-%m-%d_%H.%M")

stimDurArr = np.ones(NR_JITTERS)
for i, stimDur in enumerate(STIM_DURS[1:]):
    for n in range(NR_JITTERS):
        stimDurArr = np.append(stimDurArr, stimDur)

restDurArr = np.ones(NR_JITTERS)*REST_DURS[ITI][0]
for i, restDur in enumerate(REST_DURS[ITI][1:]):
    for n in range(NR_JITTERS):
        restDurArr = np.append(restDurArr, restDur)

# Steps between jitters
jitters = np.arange(0, NR_JITTERS)
jitters = jitters * (TR / NR_JITTERS)
jittersArr = jitters.copy()

nrStims = len(STIM_DURS)
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
    f'{FOLDER}/{sub}_{ses}_timings_TR-{TR}_jitters-{NR_JITTERS}_ITI-{ITI}.csv',
    index=False
    )

# Calculate run duration
stimDurTotal = np.sum(conditions['stimDur'])  # total stim duration
restDurTotal = np.sum(conditions['restDur'])  # total rest duration
jitterDurTotal = np.sum(conditions['jitter'])  # total rest duration

# Set initial rest (in seconds) to have a first baseline
rest_init = 30
# Set final rest (in seconds) to account for the fact that we wait for scanner
# triggers to start the stimulation on each trial. This effectively makes the
# experiment longer and is not accounted for in the calculation above.
# Also acts as baseline.
rest_end = 30

expDurtotal = stimDurTotal + restDurTotal + jitterDurTotal + rest_init + rest_end
print(f'Total experiment time: {np.round(expDurtotal/60, decimals=2)} minutes')

# Calculate nr of TRs to enter in scanning protocol
TRs = int(expDurtotal/TR)+1
print(f'We need at least {TRs} trs')
