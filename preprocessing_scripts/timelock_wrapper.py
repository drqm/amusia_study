# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:55:37 2018
Modified 02/07-2020
@author: david
"""
import timelock
import pandas as pnd

wd = 'C:/Users/au571303/Documents/projects/amusia_study'
## Parameters ##

subjects = list(range(1,35))
conditions = ['optimal','alberti','melody','hihat','familiar','unfamiliar']
std_ranges = [31,31,31,1,24,24]

data_dir  =  wd + '/data/raw/'
ica_dir =  wd + '/data/ICA/'
out_dir = wd + '/data/timelock/data/'
fig_dir = wd + '/data/timelock/figures/'

subjects_info = pnd.read_csv(wd + '/misc/subjects_info.csv', sep = ';')

tmin, tmax = -0.25, 0.4
for subject in subjects:
      idx = -1
      if not (subjects_info.bads[subjects_info.subject == subject] == 'none').bool():
          bads = list(subjects_info.bads[subjects_info.subject == subject])[0].split(",");
      else:
          bads = []
      for condition in conditions:
         idx = idx + 1
         std_range = std_ranges[idx]
         try:
            timelock.timelock(subject = subject, condition = condition,
                              std_range = std_range, data_dir = data_dir,
                              ica_dir = ica_dir, fig_dir = fig_dir,
                              out_dir = out_dir, bads = bads, tmin = tmin,
                              tmax = tmax, chan_plot = ['CPz','CP1','CP2',
                                                        'C1','C2','Cz'])
         except Exception:
             continue
            
