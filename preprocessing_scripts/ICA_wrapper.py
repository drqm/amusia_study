# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:11:48 2018
updated on  02/08/2020
@author: David R. Quiroga-Martinez
"""
import runICA
import pandas as pnd

wd = 'C:/Users/au571303/Documents/projects/amusia_study' #working directory
#subjects = [18,19]

subjects = list(range(1,35))
blocks = ['optimal','alberti','melody','familiar','unfamiliar','hihat','piazzolla','hungarian']
#blocks = ['melody']
data_path = wd + "/data/raw/"
out_path = wd + "/data/ICA/"
subjects_info = pnd.read_csv(wd + '/misc/subjects_info.csv', sep = ';')

for sub in subjects:
      subject = str(sub)
      for block in blocks:
          if not (subjects_info.bads[subjects_info.subject == sub] == 'none').bool():
              bad_channels = list(subjects_info.bads[subjects_info.subject == sub])[0].split(",");
          else:
              bad_channels = []
          try:
              runICA.runICA(subject = subject,block=block,bad_channels = bad_channels,
                    data_path = data_path,out_path = out_path)
          except Exception:
              continue 
