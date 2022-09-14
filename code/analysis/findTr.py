'''

Finds the exact times between consecutive triggers from logfiles and computes
average TR for SS-SI VASO acquisitions.

Takes logfile as input and assumes "Keypress: 5" as trigger in logfile.

'''

import re
import numpy as np

def findTR(logfile):
    with open(logfile) as f:  # Open file
        f = f.readlines()  # Get individual lines to loop through

    triggerTimes = []  # Initiate list to store trigger times

    # Loop over lines
    for line in f:
        if re.findall("Keypress: 5", line):  # Denotes triggers
            # Finds all numbers and selects the first one
            triggerTimes.append(float(re.findall("\d+\.\d+", line)[0]))

    # This depends on the way the times are set when generating the logfile.
    # In the stimulation script, the clock is reset AFTER the first trigger.
    # Therefore we are setting the first trigger time to 0 so that it matches
    # wwith following times.
    triggerTimes[0] = 0

    # Going from times to durations
    triggersSubtracted = []
    for n in range(len(triggerTimes)-1):
        triggersSubtracted.append(
                                 float(triggerTimes[n+1])
                                 -float(triggerTimes[n])
                                 )

    # As nulled and not-nulled timepoints are acquired non-symmetrically in
    # many VASO acquisitions, we can compute the mean durations between the
    # first and second trigger in a pair TR independently.
    meanFirstTriggerDur = np.mean(triggersSubtracted[::2])
    meanSecondTriggerDur = np.mean(triggersSubtracted[1::2])

    # Find mean trigger-time for a pair TR
    meanTriggerDur = (meanFirstTriggerDur+meanSecondTriggerDur)/2

    return meanTriggerDur
