# Import libraries
import pandas as pd
import contextlib
from psychopy import core, prefs, logging, event, visual, gui, monitors
from psychopy.hardware import keyboard
import numpy as np
import os
import time

### Specify TR in initial rest period through triggers
backColor = [-0.5, -0.5, -0.5]  # from -1 (black) to 1 (white)


# Load a keyboard to enable abortion.
defaultKeyboard = keyboard.Keyboard()

#***************************
#---EXPERIMENT SETTINGS
#***************************

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


# get the path that this script is in and change dir to it
_thisDir = os.path.dirname(os.path.abspath(__file__))  # get current path
os.chdir(_thisDir)  # change directory to this path


# Name and create specific subject folder
subFolderName = f"{expInfo['participant']}"
if not os.path.isdir(subFolderName):
    os.makedirs(subFolderName)

# Name and create specific session folder
sesFolderName = f"{expInfo['participant']}/{expInfo['session']}"
if not os.path.isdir(sesFolderName):
    os.makedirs(sesFolderName)


#***************************
#---PREPARE LOGFILE
#***************************

# Define a name so the log-file so it can be attributed to the
# subject/session/run.
logFileName = (
    sesFolderName
    + os.path.sep
    + f"{expInfo['participant']}"
    + f"_{expInfo['session']}"
    + f"_{expInfo['run']}"
    + '_neurovascularCoupling'
    )

# save a log file and set level for msg to be received
logFile = logging.LogFile(f'{logFileName}.log',
    level = logging.INFO
    )
# set console to receive warnings
logging.console.setLevel(logging.WARNING)

# %% MONITOR AND WINDOW
# set monitor information - CHECK WITH MPI:
distanceMon = 99  #  [99] in scanner
widthMon = 30  # [30] in scanner
PixW = 1920.0  # [1920.0] in scanner
PixH = 1200.0  # [1200.0] in psychoph lab


moni = monitors.Monitor('testMonitor', width=widthMon, distance=distanceMon)
moni.setSizePix([PixW, PixH])  # [1920.0, 1080.0] in psychoph lab

# log monitor info
logFile.write('MonitorDistance=' + str(distanceMon) + 'cm' + '\n')
logFile.write('MonitorWidth=' + str(widthMon) + 'cm' + '\n')
logFile.write('PixelWidth=' + str(PixW) + '\n')
logFile.write('PixelHeight=' + str(PixH) + '\n')

# get current date and time
dateNow = time.strftime("%Y-%m-%d_%H.%M.%S")

# set screen:
win = visual.Window(size=(PixW, PixH),
    screen = 0,
    winType='pyglet',  # winType : None, ‘pyglet’, ‘pygame’
    allowGUI=False,
    allowStencil=False,
    fullscr=True,  # for psychoph lab: fullscr = True
    monitor=moni,
    color=backColor,
    colorSpace='rgb',
    units='deg',
    blendMode='avg',
    )
# fixation dot - center [black]
dotFix = visual.Circle(
    win,
    autoLog=False,
    name='dotFix',
    units='deg',
    radius=0.075,
    fillColor=[0.0, 0.0, 0.0],
    lineColor=[0.0, 0.0, 0.0],
    )

# fixation dot - surround [yellow]
dotFixSurround = visual.Circle(
    win,
    autoLog=False,
    name='dotFix',
    units='deg',
    radius=0.15,
    fillColor=[0.5, 0.5, 0.0],
    lineColor=[0.0, 0.0, 0.0],
    )


triggerText = visual.TextStim(
    win=win,
    color='white',
    height=0.5,
    text='Experiment will start soon. Waiting for scanner'
    )



msg = visual.TextStim(win,
    text=triggerText,
    color=(1,1,1),
    height=50,
    units='pix'
    )

################### ################### ################
################### Initialize Visual ##################
################### ################### ################

tapImages = []
for i in range(0,2):
    tapImages.append(visual.ImageStim(win, pos=[0,0],
        name=f'Movie Frame {i}',
        image=f'visual_{i}.png',
        units='pix'
        )
        )

###########################################################
################### Initialize timings ####################
###########################################################


stimTimes = [1, 2, 4, 12, 24]
resttimes = [10, 12, 16, 20, 24]
trialCounter = 0

targetTimes = [5, 10, 6, 8]
targetCounter = 0

detectedTargets = 0

#########################################################
################## Start of Experiment ##################
#########################################################


nTR= 0; # total TR counter
nTR1 = 0; # even TR counter = BOLD
nTR2 = 0; # odd TR counter = VASO


globalTime = core.Clock()
fmriTime = core.Clock()
logging.setDefaultClock(fmriTime)
trialTime = core.Clock()
restTime = core.Clock()
visStimTime = core.Clock()

# timers for the targets
targetTime = core.Clock()
responseTimer = core.Clock()

msg.draw()
win.flip()
runExp = False


# Waiting for scanner and start at first trigger
event.waitKeys(
    keyList=["5"],
    timeStamped=False
    )
runExp = True
fmriTime.reset()
nTR= nTR+1; # total TR counter
nTR1 = nTR1+1; # even TR counter = VASO
nTR2 = 0; # uneven TR counter = BOLD

logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))

logging.data('StartOfRun' + str(expInfo['run']))
logging.data('StartOfParadigm')


# draw initial fixation dot
dotFixSurround.draw()
dotFix.draw()
win.flip()


# start with 3 s baseline
trialTime.reset()
logging.data('Initial baseline' + '\n')


while trialTime.getTime() < 3:
    # handle key presses each frame
    for keys in event.getKeys():
        if keys[0] in ['escape', 'q']:
            win.close()
            core.quit()
        elif keys in ['5']:
            nTR = nTR + 1
            if nTR % 2 ==1: # odd TRs
                nTR1 = nTR1 + 1

            elif nTR % 2 == 0:
                nTR2 = nTR2 +1

            logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))

targetTime.reset()



# initialize bool for target presentation
targetSwitch = False
responseSwitch = False

while runExp:
        
    if (targetCounter < len(targetTimes)) & (not targetSwitch):
        
        if targetTime.getTime() >= targetTimes[targetCounter]:
            targetCounter = targetCounter + 1

            dotFixSurround.fillColor = [1.0, 0.0, 0.0]
            dotFixSurround.lineColor = [1.0, 0.0, 0.0]
            logging.data('Target presented')
            responseTimer.reset()
            targetSwitch = True
            responseSwitch = True # enable target detection
            
    if (targetCounter > 0) & targetSwitch:
        
        if targetTime.getTime() >= (targetTimes[targetCounter-1]+1):
            
            dotFixSurround.fillColor = [0.5, 0.5, 0.0]
            dotFixSurround.lineColor = [0.0, 0.0, 0.0]
            
            targetTime.reset()
            targetSwitch = False
            
            responseTimer.reset()

            
            
            
    # display fixation
    dotFixSurround.draw()
    dotFix.draw()
    win.flip()
#    
#    stimDur = timings[key][0]
#    restDur = timings[key][1]
#    trialTime.reset()
#    
#    # Start with Stimulation
#    logFile.write('\n')
#    logging.data('stimulation' + '\n')
#    logging.data(f'stimulation started')
#    visStimType = 0
#    
#
#        if visStimTime.getTime() >= 1/8:
#            if visStimType == 0:
#                # display stimulus frame
#                tapImages[visStimType].draw()
#                # display fixation
#                dotFixSurround.draw()
#                dotFix.draw()
#                win.flip()
#                visStimTime.reset()
#                visStimType = 1
#
#        if visStimTime.getTime() >= 1/8:
#            if visStimType == 1:
#                # display stimulus frame
#                tapImages[visStimType].draw()
#                # display fixation
#                dotFixSurround.draw()
#                dotFix.draw()
#                win.flip()
#                visStimTime.reset()
#                visStimType = 0
#

    # handle key presses each frame
    for keys in event.getKeys():
        if keys[0] in ['escape', 'q']:
            win.close()
            core.quit()
        
        elif keys in ['5']:
            nTR = nTR + 1
            if nTR % 2 ==1: # odd TRs
                nTR1 = nTR1 + 1
    
            elif nTR % 2 == 0:
                nTR2 = nTR2 +1
    
            logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))
        
        elif keys in ['1']:
            
            logging.data('Key1 pressed')
            
            if (responseTimer.getTime() <= 2) & (responseSwitch):
                logging.data('Target detected')
                detectedTargets = detectedTargets + 1
                responseSwitch = False
            

    
#    logging.data(f'stimulation should stop' + '\n')
    
    
#    logging.data(f'visual stimulation stopped' + '\n')

#    trialTime.reset()
#    while restDur > trialTime.getTime():
#        # display fixation
#        dotFixSurround.draw()
#        dotFix.draw()
#        win.flip()

    if fmriTime.getTime() >= 45:
        runExp = False
    
logging.data('EndOfRun' + str(expInfo['run']) + '\n')



# %%  TARGET DETECTION RESULTS
# calculate target detection results

# detection ratio
DetectRatio = detectedTargets/len(targetTimes)
logging.data('RatioOfDetectedTargets' + str(DetectRatio))

# display target detection results to participant
resultText = 'You have detected %i out of %i targets.' % (detectedTargets,
                                                          len(targetTimes))

print(resultText)
logging.data(resultText)
# also display a motivational slogan
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
