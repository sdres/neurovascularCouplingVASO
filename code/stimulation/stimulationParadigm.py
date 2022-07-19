# Import libraries
import pandas as pd
import contextlib
from psychopy import sound, core, prefs, logging, event, visual, gui
from psychopy.hardware import keyboard
import numpy as np
import os
import time

# Load a keyboard to enable abortion.
defaultKeyboard = keyboard.Keyboard()

#***************************
#---EXPERIMENT SETTINGS
#***************************

# Set initial values
expName = 'neurovascularCouplingVASO'
expInfo = {'participant': 'sub-xx',
           'session': 'ses-00x',
           'run': 'run-0x'}

# Load a GUI in which the preset parameters can be changed.
dlg = gui.DlgFromDict(dictionary=expInfo,
    sortKeys=False,
    title=expName
    )

if dlg.OK == False:
     core.quit()  # Abort if user pressed cancel

#***************************
#---PREPARE LOGFILE
#***************************

# Define a name so the log-file so it can be attributed to the
# subject/session/run.
logFileName = f'{expInfo['participant']}'
    + '_{expInfo['session']}'
    + '_{expInfo['run']}'
    + 'neurovascularCoupling'

# save a log file and set level for msg to be received
logFile = logging.LogFile(f'{logFileName}.log',
    level = logging.INFO
    )

# set console to receive warnings
logging.console.setLevel(logging.WARNING)

# get current date and time
dateNow = time.strftime("%Y-%m-%d_%H.%M.%S")

logFile.write(
    '###############################################'
    + f'\nTHIS EXPERIMENT WAS STARTET {dateNow}\n'
    + '###############################################\n')


# ****************************************
#-----INITIALIZE GENERIC COMPONENTS
# ****************************************

# Setup the Window
win = visual.Window(
    size=[1920, 1200],
    fullscr=True,
    screen=0,
    winType='pyglet',
    allowGUI=False,
    allowStencil=False,
    monitor='testMonitor',
    color=[0,0,0],
    colorSpace='rgb',
    blendMode='avg',
    useFBO=True,
    units='height'
    )


# Initialize text
instructionsText = 'You will see flickering checkboards of varying duration.'
    + '\n'
    + 'Please maintain your gaze on the cross in the center of the screen'
    + '\n'
    + 'The cross will change its color from black to red from time to time'
    + 'When you notice a color-change, press a button on the reponse box.'
    + 'Your performance will be recorded.'
    + '\n'
    + 'Please remain as still as possible.'

msg = visual.TextStim(win,
    text=instructionsText,
    color=(1,1,1),
    height=50,
    units='pix'
    )

fixationCross = visual.TextStim(win=win,
    text='+',
    color='black',
    name='fixationCross'
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

timings = {
    '1': [1,10],
    '2': [2,12],
    '4': [4,16],
    '12': [12,20],
    '24': [24,24]
    }

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


msg.draw()
win.flip()


# Waiting for scanner and start at first trigger
event.waitKeys(
    keyList=["5"],
    timeStamped=False
    )

fmriTime.reset()
nTR= nTR+1; # total TR counter
nTR1 = nTR1+1; # even TR counter = VASO
nTR2 = 0; # uneven TR counter = BOLD

logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))

logging.data('StartOfRun' + str(expInfo['run']))
logging.data('StartOfParadigm')



fixationCross.draw()
win.flip()


# start with 30 s baseline
trialTime.reset()
logging.data('Initial baseline' + '\n')


while trialTime.getTime() < 30:
    # handle key presses each frame
    for keys in event.getKeys():
        if keys[0] in ['escape', 'q']:
            myWin.close()
            core.quit()
        elif keys in ['5']:
            nTR = nTR + 1
            if nTR % 2 ==1: # odd TRs
                nTR1 = nTR1 + 1

            elif nTR % 2 == 0:
                nTR2 = nTR2 +1

            logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))


# start with trials - here we first code one trial with a duration of 1 second
stimDur = 1
trialTime.reset()


    # Start with Stimulation
    trialTime.reset()
    logFile.write('\n')
    logging.data('stimulation' + '\n')
    logging.data(f'stimulation started')

    while stimDur > trialTime.getTime():

        if visStimTime.getTime() >= 1/16:
            if visStimType == 0:
                tapImages[visStimType].draw()
                win.flip()
                visStimTime.reset()
                visStimType = 1
                if visStimCount == 0:
                    #logging.data('visual stimulation started')
                    visStimCount = visStimCount + 1
        if visStimTime.getTime() >= 1/16:
            if visStimType == 1:
                tapImages[visStimType].draw()
                win.flip()
                visStimTime.reset()
                visStimType = 0
                if visStimCount == 0:
                    #logging.data('visual stimulation started')
                    visStimCount = visStimCount + 1

        # handle key presses each frame
        for keys in event.getKeys():
            if keys[0] in ['escape', 'q']:
                myWin.close()
                core.quit()
            elif keys in ['5']:
                nTR = nTR + 1
                if nTR % 2 ==1: # odd TRs
                    nTR1 = nTR1 + 1

                elif nTR % 2 == 0:
                    nTR2 = nTR2 +1

                logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))

    #soundStim.stop()
    logging.data(f'{tmp_choice} stimulation should stop' + '\n')
    if tmp_choice == 'visiotactile':
        sd.stop()
        #logging.data('tactile stimulation stopped' + '\n')
    fixationCross.draw()
    win.flip()
    logging.data(f'{tmp_choice} visual stimulation stopped' + '\n')
    trialCount = trialCount + 1

    trialTime.reset()
    visStimCount = 0

# End with rest until run is over
logging.data('rest' + '\n')
while 750 > fmriTime.getTime():
    # handle key presses each frame
    for keys in event.getKeys():
        if keys[0] in ['escape', 'q']:
            myWin.close()
            core.quit()
        elif keys in ['t']:
            nTR = nTR + 1
            if nTR % 2 ==1: # odd TRs
                nTR1 = nTR1 + 1

            elif nTR % 2 == 0:
                nTR2 = nTR2 +1

            logging.data('TR ' + str(nTR) + ' | TR1 ' + str(nTR1) + ' | TR2 ' + str(nTR2))




logging.data('EndOfRun' + str(expInfo['run']) + '\n')

win.close()
core.quit()
