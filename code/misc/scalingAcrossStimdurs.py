"""Scaling stimulus responses across durations"""

import os
import glob
import nibabel as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/results'

SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
# SUBS = ['sub-08']

STIMDURS = [1, 2, 4, 12, 24]

EVENTDURS = {'shortITI': np.array([11, 14, 18, 32, 48]),
             'longITI': np.array([21, 24, 28, 42, 64])}

MODALITIES = ['vaso', 'bold']

data = pd.read_csv(f'{ROOT}/ERAs_rawPlusZscored_equalizedNormalized.csv', sep=',')
plt.style.use('dark_background')



# Extract data

for j, modality in enumerate(['bold', 'vaso']):
    for i, layer in enumerate(['deep', 'middle', 'superficial']):
        # for k, stimDur in enumerate(data['stimDur'].unique()):
        for k, stimDur in enumerate([1]):
            tmp = data.loc[(data['modality'] == modality)
                               & (data['layer'] == i+1)
                               & (data['stimDur'] == stimDur)
                               & (data['dataType'] == 'raw')]

modality = 'vaso'

onesec = data.loc[(data['modality'] == modality)
               & (data['layer'] == 3)
               & (data['stimDur'] == 1)
               & (data['dataType'] == 'raw')]

twosec = data.loc[(data['modality'] == modality)
               & (data['layer'] == 3)
               & (data['stimDur'] == 2)
               & (data['dataType'] == 'raw')]

foursec = data.loc[(data['modality'] == modality)
               & (data['layer'] == 3)
               & (data['stimDur'] == 4)
               & (data['dataType'] == 'raw')]


twelvesec = data.loc[(data['modality'] == modality)
               & (data['layer'] == 3)
               & (data['stimDur'] == 12)
               & (data['dataType'] == 'raw')]

twofoursec = data.loc[(data['modality'] == modality)
               & (data['layer'] == 3)
               & (data['stimDur'] == 24)
               & (data['dataType'] == 'raw')]

vals1 = []
for volume in onesec['volume'].unique():
    tmp = onesec.loc[(onesec['volume'] == volume)]
    mean = np.mean(tmp['data'].to_numpy())
    vals1.append(mean)

vals2 = []
for volume in twosec['volume'].unique():
    tmp = twosec.loc[(twosec['volume'] == volume)]
    mean = np.mean(tmp['data'].to_numpy())
    vals2.append(mean)

vals4 = []
for volume in foursec['volume'].unique():
    tmp = foursec.loc[(foursec['volume'] == volume)]
    mean = np.mean(tmp['data'].to_numpy())
    vals4.append(mean)

vals12 = []
for volume in twelvesec['volume'].unique():
    tmp = twelvesec.loc[(twelvesec['volume'] == volume)]
    mean = np.mean(tmp['data'].to_numpy())
    vals12.append(mean)

vals24 = []
for volume in twofoursec['volume'].unique():
    tmp = twofoursec.loc[(twofoursec['volume'] == volume)]
    mean = np.mean(tmp['data'].to_numpy())
    vals24.append(mean)

test2 = np.array(vals1)*1.4
test4 = np.array(vals1)*2
test12 = np.array(vals1)*3.5
test24 = np.array(vals1)*4

x1 = np.arange(len(vals1))
x2 = np.arange(len(vals2))
x4 = np.arange(len(vals4))
x12 = np.arange(len(vals12))
x24 = np.arange(len(vals24))

plt.plot(x1, vals1, label='1')
plt.plot(x2, vals2, label='2')
plt.plot(x4, vals4, label='4')
plt.plot(x12, vals12, label='12')
plt.plot(x24, vals24, label='24')

plt.plot(x1*1.1, test2, label='1mod2')
plt.plot(x1*1.4, test4, label='1mod4')
plt.plot(x1*2.3, test12, label='1mod12')
plt.plot(x1*2.7, test24, label='1mod24')

plt.legend()
plt.show()


vals1add = np.insert(vals1, 0, 0, axis=0)
vals1add = np.insert(vals1, 0, 0, axis=0)


len(vals1add)
len(valsadd1)
valsadd1 = np.insert(vals1, -1, 0, axis=0)

new = np.add(vals1add, valsadd1)

plt.plot(x2, vals2, label='2')
plt.plot(range(len(new)), new, label='predicted')
plt.legend()
plt.show()

