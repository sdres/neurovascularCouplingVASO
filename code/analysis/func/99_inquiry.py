'''

There are some problems with the data (of course). So here, I am trying to
find out what is going wrong

'''
import pandas as pd
import glob
import os
import re
import math
import matplotlib.pyplot as plt



# Check whether triggers were timed correctly

subs = ['sub-06']
subs = ['sub-05','sub-06','sub-07','sub-08','sub-09']

ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n


subList = []
runList = []
triggerList = []

# open logfile
for sub in subs:  # Loop over participants

    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-stimulation_run-*_part-mag_bold.nii.gz'))

    for runFile in runs:
        base = os.path.basename(runFile).rsplit('.', 2)[0]
        print(f'Processing {base}')
        for i in range(1,99):
            if f'ses-0{i}' in base:
                ses = f'ses-0{i}'
        for i in range(1,99):
            if f'run-0{i}' in base:
                runNr = f'run-0{i}'

        log = f'code/stimulation/{sub}/{ses}/{sub}_{ses}_{runNr}_neurovascularCoupling.log'
        logFile = pd.read_csv(log,usecols=[0])

        # Because the column definition will get hickups if empty colums are
        # present, we find line with first trigger to then load the file anew,
        # starting with that line
        for index, row in logFile.iterrows():
            if re.search('Keypress: 5', str(row)):
                firstVolRow = index
                break

        # Define column names
        ColNames = ['startTime', 'type', 'event']

        # load logfile again, starting with first trigger
        logFile = pd.read_csv(log, sep = '\t',skiprows=firstVolRow, names = ColNames)

        tmpTriggers = []

        # Loop over lines of log and find stimulation start and stop times
        for index, row in logFile.iterrows():
            if not logFile['event'][index] != logFile['event'][index]:

                if re.search('TR1', logFile['event'][index]):
                    time = logFile['startTime'][index]
                    tmpTriggers.append(time)

        tmpTriggers = [truncate(tmpTriggers[i]-tmpTriggers[i-1],3) for i in range(1,len(tmpTriggers))]
        for i, time in enumerate(tmpTriggers):

            # if time < 1.55 or time > 1.6:
            #     print(f'deviating trigger: {base} trigger: {i}')

            subList.append(sub)
            runList.append(f'{sub}_{ses}_{runNr}')
            triggerList.append(time)

plt.hist(triggerList,100)
plt.ylim(0,4)



from collections import Counter
Counter(actualjitters)


# Check whether stimuli were jittered correctly

subList = []
runList = []
jitterList = []

for sub in subs:  # Loop over participants

    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-stimulation_run-*_part-mag_bold.nii.gz'))

    for runFile in runs:
        base = os.path.basename(runFile).rsplit('.', 2)[0]
        print(f'Processing {base}')
        for i in range(1,99):
            if f'ses-0{i}' in base:
                ses = f'ses-0{i}'
        for i in range(1,99):
            if f'run-0{i}' in base:
                runNr = f'run-0{i}'

        log = f'code/stimulation/{sub}/{ses}/{sub}_{ses}_{runNr}_neurovascularCoupling.log'
        logFile = pd.read_csv(log,usecols=[0])

        # Because the column definition will get hickups if empty colums are
        # present, we find line with first trigger to then load the file anew,
        # starting with that line
        for index, row in logFile.iterrows():
            if re.search('Keypress: 5', str(row)):
                firstVolRow = index
                break

        # Define column names
        ColNames = ['startTime', 'type', 'event']

        # load logfile again, starting with first trigger
        logFile = pd.read_csv(log, sep = '\t',skiprows=firstVolRow, names = ColNames)

        tmpJitters = []
        actualjitters = []

        # Loop over lines of log and find stimulation start and stop times
        for index, row in logFile.iterrows():
            if not logFile['event'][index] != logFile['event'][index]:


                if re.search('jitter', logFile['event'][index]):
                    currJitter = float(re.findall(r"\d+\.\d+", logFile['event'][index])[0])
                    tmpJitters.append(currJitter)

                    tmpTime = logFile['startTime'][index]

                if re.search('stimulation started', logFile['event'][index]):
                    actualJitter = logFile['startTime'][index] - tmpTime
                    actualjitters.append(actualJitter)

        actualjitters = [truncate(i,3) for i in actualjitters]

        for i, time in enumerate(actualjitters):


            subList.append(sub)
            runList.append(f'{sub}_{ses}_{runNr}')
            jitterList.append(time)

jitters = [0, 0.785, 1.570, 2.355] [0, 13/14, 12, 13]
plt.bar(test.keys(), test.values())

test = Counter(jitterList)

keys = []
for key in test:
    keys.append(key)

keys.sort()



# Time between stimulus onset and next volume

subList = []
runList = []
jitterList = []
timeToNextVolList = []

for sub in subs:  # Loop over participants

    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-0*/func/{sub}_ses-0*_task-stimulation_run-*_part-mag_bold.nii.gz'))

    for runFile in runs:
        base = os.path.basename(runFile).rsplit('.', 2)[0]
        # print(f'Processing {base}')
        for i in range(1,99):
            if f'ses-0{i}' in base:
                ses = f'ses-0{i}'
        for i in range(1,99):
            if f'run-0{i}' in base:
                runNr = f'run-0{i}'

        log = f'code/stimulation/{sub}/{ses}/{sub}_{ses}_{runNr}_neurovascularCoupling.log'
        logFile = pd.read_csv(log,usecols=[0])

        # Because the column definition will get hickups if empty colums are
        # present, we find line with first trigger to then load the file anew,
        # starting with that line
        for index, row in logFile.iterrows():
            if re.search('Keypress: 5', str(row)):
                firstVolRow = index
                break

        # Define column names
        ColNames = ['startTime', 'type', 'event']

        # load logfile again, starting with first trigger
        logFile = pd.read_csv(log, sep = '\t',skiprows=firstVolRow, names = ColNames)

        tmpJitters = []
        actualjitters = []
        tmpTimeToVol = []
        counter = 0

        stimSwitch = False
        # Loop over lines of log and find stimulation start and stop times
        for index, row in logFile.iterrows():
            if not logFile['event'][index] != logFile['event'][index]:


                if re.search('jitter', logFile['event'][index]):
                    currJitter = float(re.findall(r"\d+\.\d+", logFile['event'][index])[0])
                    tmpJitters.append(currJitter)
                    tmpTime = logFile['startTime'][index]

                if re.search('stimulation started', logFile['event'][index]):
                    stimSwitch = True
                    actualJitter = logFile['startTime'][index] - tmpTime
                    actualjitters.append(actualJitter)
                    tmpTime = logFile['startTime'][index]

                if stimSwitch:

                    if currJitter < 2 and currJitter !=
                        if re.search('Keypress: 5', logFile['event'][index]):
                            volStart = logFile['startTime'][index]
                            timeToVol = volStart - tmpTime
                            tmpTimeToVol.append(timeToVol)
                            stimSwitch = False

                    if currJitter > 2:
                        if re.search('Keypress: 5', logFile['event'][index]):
                            counter += 1
                            if counter >=2:
                                volStart = logFile['startTime'][index]
                                timeToVol = volStart - tmpTime
                                tmpTimeToVol.append(timeToVol)
                                stimSwitch = False
                                counter = 0


        tmpTimeToVol = [truncate(i,3) for i in tmpTimeToVol]

        for i, time in enumerate(tmpTimeToVol):

            subList.append(sub)
            runList.append(f'{sub}_{ses}_{runNr}')
            jitterList.append(tmpJitters[i])
            timeToNextVolList.append(time)



Counter(timeToNextVolList)


import pandas as pd
import seaborn as sns
import numpy as np
data = pd.DataFrame({'jitter':jitterList, 'delay':timeToNextVolList})

sns.histplot(data=data.query('jitter==1.570'), x="delay")

boldJitters = {0.0: 1.570, 0.785: 0.785, 1.57: 0, 2.355: 2.355}

for jit in [0, 0.785, 1.570, 2.355]:
    delay = np.mean(data.query('jitter==@jit')['delay'])
    diff = delay-boldJitters[jit]
    print(diff)


plt.ylim(0,3)


plt.scatter(jitterList,timeToNextVolList)
plt.xlabel('jitter')



jitters = [0, 0.785, 1.570, 2.355] [0, 13/14, 12, 13]
plt.bar(test.keys(), test.values())




with open(log) as f:  # Open file
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

0.785*2 - 1.5383090909090917
meanFirstTriggerDur - 1.5383090909090917
