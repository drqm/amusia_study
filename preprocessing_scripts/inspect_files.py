# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:36:01 2019

@author: au571303
"""

import mne

data_path = 'C:/Users/dariq/Dropbox/PC/Documents/projects/amusia_study/data/raw/'
subject = '2'
blocks = ['optimal','alberti','melody','familiar',
          'unfamiliar','hihat','piazzolla','hungarian']
block = blocks[7]
file_name = data_path + subject + '_' + block + '.bdf'
raw = mne.io.read_raw_bdf(file_name,preload=True)
events = mne.find_events(raw, min_duration = 0.001,initial_event=True)
fig = raw.plot(events = events, lowpass = 35)

#fig = raw.plot(show = False)
#fig.set_size_inches(14,7)
