"""Get the dates of the functional scans"""

import os
import glob
import datetime
import calendar
import time
import numpy as np

subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']

root = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation'
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti'

daysapart = []

for sub in subs:
    print(sub)
    # =========================================================================
    # Look for sessions
    # Collect all runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*_part-mag*.nii.gz'))

    # Initialte list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1, 6):  # We had a maximum of 2 sessions
            if f'ses-0{i}' in run:
                sessions.append(f'ses-0{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))

    dates = []

    for ses in sessions:
        run = sorted(glob.glob(f'{root}/{sub}/{ses}/*run-01*.log'))[0]

        # Both the variables would contain time
        # elapsed since EPOCH in float
        ti_c = os.path.getctime(run)

        # Converting the time in seconds to a timestamp
        c_ti = time.ctime(ti_c)
        print(c_ti)

        # file creation timestamp in float
        c_time = os.path.getmtime(run)
        # convert creation timestamp into DateTime object
        dt_c = datetime.datetime.fromtimestamp(c_time)

        day = dt_c.day
        month = dt_c.month
        year = dt_c.year

        dates.append([day, month])

    print(dates)
    for i, date in enumerate(dates[:-1]):
        currMonth = date[1]
        nextMonth = dates[i + 1][1]

        currDate = date[0]
        nextDate = dates[i+1][0]

        if currMonth < nextMonth:
            # Get length of current month
            lenMonth = calendar.monthrange(2022, currMonth)[1]

            daysRemaining = lenMonth - currDate

            diff = daysRemaining + nextDate

        if currMonth == nextMonth:

            diff = nextDate - currDate

        daysapart.append(diff)


np.mean(daysapart)
np.std(daysapart)
np.min(daysapart)
np.max(daysapart)