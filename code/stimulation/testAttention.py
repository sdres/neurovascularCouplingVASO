'''
Created on Thu Jul 21 17:47:28 2022

@author: Sebastian.Dresbach
'''


from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import visual, event, core, monitors, logging, gui, data, misc
import numpy as np
import os
from psychopy.hardware import keyboard


### Specify TR in initial rest period through triggers
backColor = [-0.5, -0.5, -0.5]  # from -1 (black) to 1 (white)


# Load a keyboard to enable abortion.
defaultKeyboard = keyboard.Keyboard()

# %% MONITOR AND WINDOW
# set monitor information - CHECK WITH MPI:
distanceMon = 99  #  [99] in scanner
widthMon = 30  # [30] in scanner
PixW = 1920.0  # [1920.0] in scanner
PixH = 1200.0  # [1200.0] in psychoph lab


moni = monitors.Monitor('testMonitor', width=widthMon, distance=distanceMon)
moni.setSizePix([PixW, PixH])  # [1920.0, 1080.0] in psychoph lab

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

# fixation dot - surround [red]
dotFixSurroundTarget = visual.Circle(
    win,
    autoLog=False,
    name='dotFix',
    units='deg',
    radius=0.15,
    fillColor=[1.0, 0.0, 0.0],
    lineColor=[1.0, 0.0, 0.0],
    )
    
triggerText = visual.TextStim(
    win=win,
    color='white',
    height=0.5,
    text='Experiment will start soon. Waiting for scanner'
    )


# give the system time to settle
core.wait(1)

# wait for scanner trigger
triggerText.draw()
win.flip()
core.wait(1)
# event.waitKeys(keyList=['5'], timeStamped=False)

dotFixSurround.draw()
dotFix.draw()
win.flip()

core.wait(3)

# present target for 0.5 seconds
dotFixSurroundTarget.draw()
dotFix.draw()
win.flip()
core.wait(0.5)

# present normal fix again
dotFixSurround.draw()
dotFix.draw()
win.flip()

core.wait(4)

# %% FINISH
core.quit()
