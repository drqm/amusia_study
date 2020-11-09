import mne
import matplotlib.pyplot as plt
import pandas as pnd
import numpy as np
import matplotlib.gridspec as gridspec
import pickle 
import matplotlib as mpl
import os

wd = 'C:/Users/au571303/Documents/projects/amusia_study'
os.chdir(wd + '/study2/scripts')

subjects = pnd.read_csv(wd +  '/misc/subjects_info.csv', sep = ';')
#subjects = subjects_info[['subject','group']]

group_counts = {'amusic': 0, 'control': 0, 'all': 0}
all_groups = {'controls': {}, 'amusics': {}, 'all': {}}
all_stats = {'controls': {}, 'amusics': {}, 'all': {}}
sub_mapping = {'controls': {}, 'amusics': {}, 'all': {}}

data_dir = wd + '/data/timelock/data/'
out_dir = wd + '/study2/results/'
############################ Load the data ###################################
conds = ['optimal' , 'hihat']
for idx, row in subjects.iterrows():
    print(idx, 'loading subject:',row['subject'], '   group:', row['group'])
    sub, group = row['subject'], row['group']
    group_counts[group] = group_counts[group] + 1
    group_counts['all'] = group_counts['all'] + 1
    for cond in conds:
      filename = data_dir + str(sub) + '_' + cond + '_timelocked-ave.fif'
      evokeds = mne.read_evokeds(filename)
      if  group_counts[group] == 1:
               all_groups[group + 's'][cond] = {}
               all_stats[group + 's'][cond] = {}
               sub_mapping[group + 's'][cond] = {}
      if  group_counts['all'] == 1:
               all_groups['all'][cond] = {}
               all_stats['all'][cond] = {}
               sub_mapping['all'][cond] = {}
      for evkd in evokeds:        
        feat_name = evkd.comment 
        if not feat_name in all_groups[group + 's'][cond]:
          all_groups[group + 's'][cond][feat_name] = []
          new_mapping = np.array([[sub,0]])
          sub_mapping[group + 's'][cond][feat_name] = new_mapping
        else:
          old_mapping = sub_mapping[group + 's'][cond][feat_name] 
          new_mapping = np.array([[sub,len(all_groups[group + 's'][cond][feat_name])]])
          sub_mapping[group + 's'][cond][feat_name] = np.concatenate([old_mapping,
                                                                new_mapping])
        if not feat_name in all_groups['all'][cond]:
          all_groups['all'][cond][feat_name] = []
          new_mapping = np.array([[sub,0]])
          sub_mapping['all'][cond][feat_name] = new_mapping
        else:
          old_mapping = sub_mapping['all'][cond][feat_name] 
          new_mapping = np.array([[sub,len(all_groups['all'][cond][feat_name])]])
          sub_mapping['all'][cond][feat_name] = np.concatenate([old_mapping,
                                                                new_mapping])
            
        all_groups[group + 's'][cond][feat_name].append(evkd)
        all_groups['all'][cond][feat_name].append(evkd)

        cur_data = evkd.data.copy()
        cur_data = np.transpose(cur_data,(1,0)) # first copy and transpose data
        nrow, ncol = cur_data.shape
        cur_data = np.reshape(cur_data, (1,nrow,ncol)) #reshape for concatenation
        if not feat_name in all_stats[group + 's'][cond]:
          all_stats[group + 's'][cond][feat_name] = cur_data.copy()
        else:
          # concatenate along first dimension:
          stats_array = np.concatenate((all_stats[group + 's'][cond][feat_name],
                                        cur_data.copy()),0)
          all_stats[group + 's'][cond][feat_name] = stats_array.copy() 
          
        if not feat_name in all_stats['all'][cond]:
          all_stats['all'][cond][feat_name] = cur_data.copy()
        else:
          # concatenate along first dimension:
          stats_array = np.concatenate((all_stats['all'][cond][feat_name],
                                        cur_data.copy()),0)
          all_stats['all'][cond][feat_name] = stats_array.copy()

########################### Grand averages ##################################
grand_avg = {'controls': {}, 'amusics': {}, 'all': {}}
for group in all_groups:
    for grand_cond in all_groups[group]:
        grand_avg[group][grand_cond] = {}
        for feat in all_groups[group][grand_cond]:
            grand_prov = mne.grand_average(all_groups[group][grand_cond][feat],
                                           drop_bads = True)
            grand_prov.comment = feat
            # correct the scaling with stats data
            grand_prov.data = np.transpose(np.mean(all_stats[group][grand_cond][feat],0),(1,0))
            grand_avg[group][grand_cond][feat] = grand_prov
            print(grand_prov)

# fileavg = open(out_dir + 'grand_avg.p','wb')
# pickle.dump(grand_avg,fileavg)
# fileavg.close()

######################## Extract Mean Amplitudes ############################
components = ['MMN','P3a']
TWs = [[0.07,0.25],[0.2,0.4]]
flip = [1,-1]
chan_sels = [['Fz','F1','F2','F3','F4','FCz','FC1','FC2','FC3','FC4'],
            ['FCz','FC1','FC2','FC3','FC4','Cz','C1','C2','C3','C4']]
times = grand_avg['controls']['optimal']['pitch_MMN'].times

features = ['intensity','timbre','location']
conditions = ['optimal','hihat']
MA_data = pnd.DataFrame() # pandas df to store the averages

for idx, row in subjects.iterrows():
  sgroup = row['group'] + 's' # fetch subject's group
  scode = row['subject'] # fetch subject's code
  mbea = row['Mean MBEA']
  mbea_pitch = row['Mean MBEA Pitch']
  PDT = row['PDT'] #pitch discrimnation threshold
  age = row['age']
  for c in conditions:
    for f in features:
        # map s code to s index in the evoked arrays
        smap = sub_mapping[sgroup][c][f + '_MMN'] 
        sidx = np.where(smap[:,0] == scode)[0]               
        # only if the subject's code was found, extract MA:
        if sidx.size > 0:
            cur_grand = grand_avg[sgroup][c][f + '_MMN'].copy()
            cur_evkd = all_groups[sgroup][c][f + '_MMN'][sidx[0]].copy()
            cur_std = all_groups[sgroup][c]['standard'][sidx[0]].copy()
            cur_dev = all_groups[sgroup][c][f][sidx[0]].copy()
            crow_dict = {"subject": [scode], 
                            "group": sgroup,
                            "age": age,
                            "MBEA": mbea,
                            "MBEA_pitch": mbea_pitch,
                            "PDT": PDT,
                            "condition": c,
                            "feature": f}
            for coidx, co in enumerate(components):
                chan_sel = chan_sels[coidx]
                t1, t2 = TWs[coidx][0],TWs[coidx][1]
                
                cur_grand.pick_channels(chan_sel)
                cur_evkd.pick_channels(chan_sel)
                cur_std.pick_channels(chan_sel)
                cur_dev.pick_channels(chan_sel)
                              
                cur_grand_data = np.mean(cur_grand.data,0)
                cur_data = np.mean(cur_evkd.data,0)
                cur_data_std = np.mean(cur_std.data,0)
                cur_data_dev = np.mean(cur_dev.data,0)
                cur_data_f = cur_data.copy()*flip[coidx]
                cur_grand_data_f = cur_grand_data.copy()*flip[coidx]
                
                ## Get latency for this subject
                peak_idx = ((cur_data_f[1:-1] - cur_data_f[2:] < 0) &
                          (cur_data_f[1:-1] - cur_data_f[0:-2] < 0) &
                          (times[1:-1] >= t1) & (times[1:-1] <= t2))
                cur_data_f = cur_data_f[1:-1]                
                peak_idx2 = np.where((peak_idx == True) &
                                    (cur_data_f == np.amin(cur_data_f[peak_idx])))
                latency = times[1:-1][peak_idx2]
                
                ## Get peak of the grand average               
                peak_idx_grand = ((cur_grand_data_f[1:-1] - cur_grand_data_f[2:] < 0) &
                                  (cur_grand_data_f[1:-1] - cur_grand_data_f[0:-2] < 0) &
                                  (times[1:-1] >= t1) & (times[1:-1] <= t2))
                cur_grand_data_f = cur_grand_data_f[1:-1]               
                peak_idx_grand2 = np.where((peak_idx_grand == True) &
                                    (cur_grand_data_f == np.amin(cur_grand_data_f[peak_idx_grand])))
                latency_grand = times[1:-1][peak_idx_grand2]
                
                ## Set time window and extract the amplitude
                tw = np.where((times[1:-1] >= latency_grand - 0.04) & 
                              (times[1:-1] <= latency_grand + 0.04))
                                
                MA = np.mean(cur_data[tw])
                MA_std = np.mean(cur_data_std[tw])
                MA_dev = np.mean(cur_data_dev[tw]) 

                latency = np.round(latency*1000)[0]
                crow_dict[co + '_amplitude'] = MA
                crow_dict[co + '_latency'] = latency
                crow_dict[co + '_standard'] = MA_std
                crow_dict[co + '_deviant'] = MA_dev
            
            currow = pnd.DataFrame(crow_dict)
            if MA_data.empty:
              MA_data = currow
            else:
              MA_data = MA_data.append(currow)

# and save to csv:
MA_data.to_csv( wd + '/data/MA/MA_toneness.csv', index = None, header = True)

