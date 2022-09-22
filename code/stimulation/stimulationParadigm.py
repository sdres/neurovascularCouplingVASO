"""Neurovascular Coupling with VASO experiment stimulation."""

import pandas as pd
import contextlib
from psychopy import core, prefs, logging, event, visual, gui, monitors
from psychopy.hardware import keyboard
import numpy as np
import os
import time

# Load a keyboard to enable abortion.
defaultKeyboard = keyboard.Keyboard()

# =============================================================================
# Initial experiment settings
# =============================================================================

# Set initial values
expName = 'neurovascularCouplingVASO'
expInfo = {'participant': 'sub-01',
           'session': 'ses-01',
           'run': 'run-01'
           }

# Load a GUI in which the preset parameters can be changed.
dlg = gui.DlgFromDict(dictionary=expInfo,
                      sortKeys=False,
                      title=expName
                      )

if dlg.OK == False:
    core.quit()  # Abort if user pressed cancel

# Change directory to where this script is
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Name and create specific subject folder
subFolderName = f"{expInfo['participant']}"
if not os.path.isdir(subFolderName):
    os.makedirs(subFolderName)

# Name and create specific session folder
sesFolderName = f"{expInfo['participant']}/{expInfo['session']}"
if not os.path.isdir(sesFolderName):
    os.makedirs(sesFolderName)

# =============================================================================
# Prepare logfile
# =============================================================================

# Define a name so the log-file so it can be attributed to subject/session/run
logFileName = (
    sesFolderName
    + os.path.sep
    + f"{expInfo['participant']}"
    + f"_{expInfo['session']}"
    + f"_{expInfo['run']}"
    + '_neurovascularCoupling'
    )

# Save a log file and set level for msg to be received
logFile = logging.LogFile(f'{logFileName}.log', level=logging.INFO)

# Set console to receive warnings
logging.console.setLevel(logging.WARNING)

# =============================================================================
# Prepare monitor and window
# =============================================================================

# Set monitor information - CHECK WITH MPI:
distanceMon = 99  # [99] in scanner
widthMon = 30  # [30] in scanner
PixW = 1920.0  # [1920.0] in scanner
PixH = 1200.0  # [1200.0] in psychoph lab

moni = monitors.Monitor('testMonitor', width=widthMon, distance=distanceMon)
moni.setSizePix([PixW, PixH])  # [1920.0, 1080.0] in psychoph lab

# Log monitor info
logFile.write('MonitorDistance=' + str(distanceMon) + 'cm' + '\n')
logFile.write('MonitorWidth=' + str(widthMon) + 'cm' + '\n')
logFile.write('PixelWidth=' + str(PixW) + '\n')
logFile.write('PixelHeight=' + str(PixH) + '\n')

# Get current date and time
dateNow = time.strftime("%Y-%m-%d_%H.%M.%S")

backColor = [-0.5, -0.5, -0.5]  # from -1 (black) to 1 (white)

# Set screen:
win = visual.Window(size=(PixW, PixH),
                    screen=0,
                    winType='pyglet',  # winType : None, ‘pyglet’, ‘pygame’
                    allowGUI=False,
                    allowStencil=False,
                    fullscr=True,  # for psychoph lab: fullscr = True
                    monitor=moni,
                    color=backColor,
                    colorSpace='rgb',
                    units='deg',
                    blendMode='avg'
                    )

# Fixation dot - center [black]
dotFix = visual.Circle(
    win,
    autoLog=False,
    name='dotFix',
    units='deg',
    radius=0.075,
    fillColor=[0.0, 0.0, 0.0],
    lineColor=[0.0, 0.0, 0.0],
    )

# Fixation dot - surround [yellow]
dotFixSurround = visual.Circle(
    win,
    autoLog=False,
    name='dotFix',
    units='deg',
    radius=0.15,
    fillColor=[0.5, 0.5, 0.0],
    lineColor=[0.0, 0.0, 0.0],
    )

# Set initial text
triggerText = visual.TextStim(
    win=win,
    color='white',
    height=0.5,
    text='Experiment will start soon. Waiting for scanner'
    )

# =============================================================================
# Initialize Visual stimulus
# =============================================================================

tapImages = []  # make list for frames
for i in range(0, 2):  # loop over frames
    tapImages.append(visual.ImageStim(  # append frame to list
        win,
        pos=[0, 0],
        name=f'Movie Frame {i}',
        image=f'visual_{i}.png',
        units='pix'
        )
        )

stimFrameRate = 8  # Set stimulus frame rate in Hz

# =============================================================================
# Initialize timings
# =============================================================================

trialTiming = pd.read_csv('/Users/sebastiandresbach/git/neurovascularCouplingVASO/code/stimulation/conditionTimings_TR-3.447_jitters-4_2022-08-19_11.07.csv')
trialCounter = 0 # set counter for trials

# Get duration of entire experiment
initialRest = 30  # set inital rest
stimDurTotal = np.sum(trialTiming['stimDur'])  # sum stimulus durations
restDurTotal = np.sum(trialTiming['restDur'])  # sum rest durations
jitDurTotal = np.sum(trialTiming['jitter'])  # sum jitter durations
# Set extra rest after experiment to account for longer run durations
# due to waiting for triggers and extra baseline
finalRest = 30

# Calculate entire run duration
expDurTotal = (
    initialRest
    + stimDurTotal
    + restDurTotal
    + jitDurTotal
    + finalRest
    )

expDurTotalOrig = expDurTotal

# Generate pseudo-random targets for attention task
minInterval = 40  # min duration between targets
maxInterval = 80  # max duration between targets
nrTargets = int(expDurTotal/60)  # present on average one target per minute

# Create array with intervcals between targets
targetTimes = np.random.uniform(minInterval, maxInterval, nrTargets)

targetDur = 0.5  # target duration in seconds
targetCounter = 0  # set counter for targets
detectedTargets = 0  # set counter for detected targets

# Set up TR counters
nTR = 0  # total TR counter
nTR1 = 0  # odd TR counter = VASO
nTR2 = 0  # even TR counter = BOLD

# =============================================================================
# Start of Experiment
# =============================================================================

# Set up clocks
fmriTime = core.Clock()
logging.setDefaultClock(fmriTime)

trialTime = core.Clock()  # timer for a trial
visStimTime = core.Clock()  # timer for visual stimulation
waitTime = core.Clock()  # Set timer to account for waiting for triggers

# Timers for the targets
targetTime = core.Clock()  # time until next target
responseTimer = core.Clock()  # time from target onset until response

triggerText.draw()
win.flip()
runExp = False


# Waiting for scanner and start at first trigger
event.waitKeys(
    keyList=["5"],
    timeStamped=False
    )

runExp = True
fmriTime.reset()  # reset time for experiment

nTR = nTR + 1  # update total TR counter
nTR1 = nTR1 + 1  # update odd (VASO) TR counter
nTR2 = 0  # even (BOLD) TR counter does not change yet

logging.data(
    'StartOfRun '
    + str(expInfo['participant'])
    + '_'
    + str(expInfo['session'])
    + '_'
    + str(expInfo['run'])
    )

logging.data(
    'TR '
    + str(nTR)
    + ' | TR1 '
    + str(nTR1)
    + ' | TR2 '
    + str(nTR2)
    )

# Draw initial fixation dot
dotFixSurround.draw()
dotFix.draw()
win.flip()

# Start with initial baseline
logging.data('Initial baseline' + '\n')
while fmriTime.getTime() < initialRest:
    # Evaluate keys
    for keys in event.getKeys():
        if keys[0] in ['escape', 'q']:
            win.close()
            core.quit()

        elif keys in ['5']:
            nTR = nTR + 1
            if nTR % 2 == 1:  # odd TRs
                nTR1 = nTR1 + 1

            elif nTR % 2 == 0:
                nTR2 = nTR2 + 1

            logging.data(
                'TR '
                + str(nTR)
                + ' | TR1 '
                + str(nTR1)
                + ' | TR2 '
                + str(nTR2)
                )

        elif keys in ['1']:  # target detection
            logging.data('Key1 pressed')

            # if button one is pressed in 2 s window after target
            # and response is possible
            if (responseTimer.getTime() <= 2) and (responseSwitch):
                # record that the target was detected
                logging.data('Target detected')
                detectedTargets = detectedTargets + 1
                # disable target counter until next target
                responseSwitch = False


# initialize bools for target presentation
targetSwitch = False  # target on/off
responseSwitch = False  # whether reponse to target was registered

# initialize bools for stimulus presentation
stimSwitch = False  # stimulation on/off
triggerSwitch = False  # wait for VASO trigger in stimulation

# trialTime.reset()  # reset trialtime after initial fixation - not necessary anymore because we reset after detected VASO trigger
targetTime.reset()  # reset target after initial fixation (no targets before)
waitTime.reset()

while runExp:
    # -------------------------------------------------------------------------
    # Attention task
    # -------------------------------------------------------------------------
    # Execute attention task only if there is targets remaining
    # and we are currently not presenting a target anyway
    if (targetCounter < len(targetTimes)) and (not targetSwitch):

        # check whether it is time for a target
        if targetTime.getTime() >= targetTimes[targetCounter]:
            # update counter for targets
            targetCounter = targetCounter + 1
            # set dot surround to red
            dotFixSurround.fillColor = [1.0, 0.0, 0.0]
            dotFixSurround.lineColor = [0.0, 0.0, 0.0]

            logging.data('Target presented')
            # reset timer because from now on the target detection window
            # is open
            responseTimer.reset()
            targetSwitch = True  # a target is being presented
            responseSwitch = True  # enable target detection

    # do not execute this before first target is being presented or when
    # there os no target at the moment
    if (targetCounter > 0) and targetSwitch:
        # if the target has been there for the target duration
        if (targetTime.getTime() >= (targetTimes[targetCounter-1] + targetDur)):

            # set dot surround back to yellow
            dotFixSurround.fillColor = [0.5, 0.5, 0.0]
            dotFixSurround.lineColor = [0.0, 0.0, 0.0]
            # reset timer to find time until next target
            targetTime.reset()
            targetSwitch = False  # no more target is being presented
            # reset timer because the target detection window
            # is still open for 2 more seconds
            responseTimer.reset()

    # -------------------------------------------------------------------------
    # Visual stimulation
    # -------------------------------------------------------------------------
    if triggerSwitch:
        # Check wether we are past the jitter time but not past the stimulation
        # duration plus the jitter time for the current trial
        if (trialTime.getTime() > jitDur and (trialTime.getTime() <= (stimDur + jitDur))):

            # If we haven't started the stimulation start now
            if not stimSwitch:
                logging.data('')
                logging.data(f'stimulation started')
                logging.data('')
                # We are starting to present a stimulus so reset the timer
                visStimTime.reset()
                # Start with frame 0
                visStimFrame = 0
                # Turn on switch that we are presenting a stimulus
                stimSwitch = True

            # We have to change stim frames at the desired rate
            if stimSwitch and visStimTime.getTime() >= 1/8:

                # Switch to other frame for next iteration
                if visStimFrame == 0:
                    visStimFrame = 1
                elif visStimFrame == 1:
                    visStimFrame = 0

                # Reset timer
                visStimTime.reset()

        # Check whether we reached the end of the stimulation
        if trialTime.getTime() > (stimDur + jitDur):
            # Turn off stimulation if yes
            if stimSwitch:
                stimSwitch = False
                logging.data('')
                logging.data(f'stimulation stopped')
                logging.data('')

        # Check whether we have reached the end of the entire trial
        # this INCLUDES the rest period
        if trialTime.getTime() > trialDur:
            trialCounter = trialCounter + 1
            logging.data(f'')
            logging.data(f'Trial complete')
            if trialCounter < (trialTiming.shape[0]-1):
                logging.data(f'Waiting for VASO trigger to start next trial')
                waitTime.reset()
            logging.data(f'')
            triggerSwitch = False

        if not stimSwitch:
            # display only fixation
            dotFixSurround.draw()
            dotFix.draw()
            win.flip()

        if stimSwitch:
            # display stimulus frame and fixation
            tapImages[visStimFrame].draw()
            dotFixSurround.draw()
            dotFix.draw()
            win.flip()

    # for fixation baseline (at end or in between trials)
    else:
        dotFixSurround.draw()
        dotFix.draw()
        win.flip()

    for keys in event.getKeys():
        if keys[0] in ['escape', 'q']:
            win.close()
            core.quit()

        elif keys in ['5']:
            nTR = nTR + 1

            if nTR % 2 == 1:  # odd TRs
                nTR1 = nTR1 + 1

                # wait for VASO trigger
                if not triggerSwitch:

                    # only execute until we reached the last trial
                    if trialCounter < trialTiming.shape[0]:
                        logging.data(f'Waited {waitTime.getTime()} seconds for trigger')
                        expDurTotal = expDurTotal + waitTime.getTime()

                        triggerSwitch = True
                        trialTime.reset()

                        # get durations of current trial
                        currTrial = trialTiming.iloc[[trialCounter]]

                        trialDur = ((
                            currTrial['stimDur']
                            + currTrial['restDur']
                            + currTrial['jitter']
                            )[trialCounter])  # We have to give it the index because dataFrame.iloc

                        stimDur = currTrial['stimDur'][trialCounter]
                        restDur = currTrial['restDur'][trialCounter]
                        jitDur = currTrial['jitter'][trialCounter]

                        # Log stimulus info of next trial
                        logging.data('')
                        logging.data(f'Trial {trialCounter+1}')
                        logging.data(f'stimDur = {stimDur}')
                        logging.data(f"jitter = {jitDur}")
                        logging.data(f'restDur = {restDur}\n')
                        logging.data('')

            elif nTR % 2 == 0:
                nTR2 = nTR2 + 1

            logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 '
                         + str(nTR2))

        elif keys in ['1']:
            logging.data('Key1 pressed')

            if (responseTimer.getTime() <= 2) and (responseSwitch):
                logging.data('Target detected')
                detectedTargets = detectedTargets + 1
                responseSwitch = False


    if fmriTime.getTime() > expDurTotal:
        runExp = False

logging.data(
    'EndOfRun '
    + str(expInfo['participant']) + '_'
    + str(expInfo['session']) + '_'
    + str(expInfo['run']) + '\n'
    )

waitTime = expDurTotal - expDurTotalOrig
logging.data(f'We spent {waitTime} seconds waiting for triggers')

# %%  TARGET DETECTION RESULTS
# calculate target detection results

# Detection ratio
DetectRatio = detectedTargets/len(targetTimes)
logging.data('RatioOfDetectedTargets ' + str(DetectRatio))

# Display target detection results to participant
resultText = 'You have detected %i out of %i targets.' % (detectedTargets,
                                                          targetCounter)

print(resultText)
logging.data(resultText)
# Also display a motivational slogan
if DetectRatio >= 0.95:
    feedbackText = 'Excellent! Keep up the good work'
elif DetectRatio < 0.95 and DetectRatio >= 0.85:
    feedbackText = 'Well done! Keep up the good work'
elif DetectRatio < 0.85 and DetectRatio >= 0.65:
    feedbackText = 'Please try to focus more'
else:
    feedbackText = 'You really need to focus more!'

targetText = visual.TextStim(
    win=win,
    color='white',
    height=0.5,
    pos=(0.0, 0.0),
    autoLog=False,
    )

targetText.setText(resultText + '\n' + feedbackText)
logFile.write(str(resultText) + '\n')
logFile.write(str(feedbackText) + '\n')
targetText.draw()
win.flip()
core.wait(5)

win.close()
core.quit()
