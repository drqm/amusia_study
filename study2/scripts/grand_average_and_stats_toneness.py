# This script loads data, calculates grand averages, 
# extracts mean amplitudes and produces plots.
# Plots include Figures 2, 3 and 4 in the paper.

# You can run the script as is or you can also run it interactively using
# VSCode or something similar. This is what '# %%' lines are for.

# %%
import mne
import matplotlib.pyplot as plt
import pandas as pnd
import numpy as np
import matplotlib.gridspec as gridspec
import pickle 
import matplotlib as mpl
import os

# %%
# place your working directory here
# wd = 'C:/Users/au571303/Documents/projects/amusia_study'
# wd = '/Users/jonathannasielski/Desktop/UCU/S6/Thesis/amusia_study'
wd = '/Users/kbas/cloud/lab/amusia_study'
os.chdir(wd + '/study2/scripts')

# load subjects data
subjects = pnd.read_csv(wd +  '/data/subjects_info.csv', sep = ',')
# subjects = subjects_info[['subject','group']]

# dicts for holding data

# how many people in each groups
group_counts = {'amusic': 0, 'control': 0, 'all': 0}

# mne evoked objects in groups
all_groups = {'controls': {}, 'amusics': {}, 'all': {}}

# raw data, not mne evoked objects
all_stats = {'controls': {}, 'amusics': {}, 'all': {}}

# mapping of subjects to raw data
sub_mapping = {'controls': {}, 'amusics': {}, 'all': {}}

# paths for storing data and output results
data_dir = wd + '/data/timelock/data/'
out_dir = wd + '/study2/results/'

# %%
############################ Load the data ###################################
conds = ['optimal' , 'hihat']

# iterate for each subject
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

# %%
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

# %%
######################## Extract Mean Amplitudes ############################
components = ['MMN','P3a']
TWs = [[0.07,0.25],[0.15,0.35]]
flip = [1,-1]
chan_sels = [['Fz','F1','F2','F3','F4','FCz','FC1','FC2','FC3','FC4'],
            ['Fz','F1','F2','F3','F4','FCz','FC1','FC2','FC3','FC4']]
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

# get ids of EEG channels as a dict with channel names as keys:
# get a single evoked object
ch_n = all_groups['all']['optimal']['intensity'][0].ch_names
chann_ids = dict(zip(ch_n, range(len(ch_n))))


# %%
# PLOTS

# Figure 3. Standards, deviants and MMN.

fig = plt.figure(figsize = (10,15))
gs = gridspec.GridSpec(8,3,left=0.05, right=0.98,
                       top=0.93,bottom = 0.05,
                       wspace=0.35, hspace = 0.35,
                       width_ratios = [0.001,1,1],
                       height_ratios = [1,1,0.1,1,1,0.1,1,1])
channs = ['Fz']
features = ['intensity', 'timbre', 'location']
conds = ['optimal', 'hihat']
groups = ['controls', 'amusics']
spac = 0.315

for fidx,f in enumerate(features):
  for gidx,g in enumerate(groups):
    group_ax = plt.subplot(gs[fidx*3+gidx,0])
    group_ax.axis('off')
    group_ax.annotate(g, xy = (0.5,0.5),
        xytext = (0.1,0.95),
        # xycoords = ('figure fraction','figure fraction'), 
        textcoords='offset points', size=13.5, ha='center',
        va='center',rotation = 90)
    
    for cidx,c in enumerate(conds):
      std = grand_avg[g][c]['standard'].copy().pick_channels(channs).data.mean(0)*(1e6)
      dev = grand_avg[g][c][f].copy().pick_channels(channs).data.mean(0)*(1e6)
      MMN = grand_avg[g][c][f+'_MMN'].copy().pick_channels(channs).data.mean(0)*(1e6)
      time = grand_avg[g][c]['standard'].times
      evkd_ax1 = plt.subplot(gs[fidx*3+gidx,cidx+1])
      evkd_ax1.plot(time,std,'b--',label = 'standard')
      evkd_ax1.plot(time,dev,'r--',label = 'deviant')
      evkd_ax1.plot(time,MMN,'k-',label = 'MMN')

      # individual subject MMN traces
      channel_id = chann_ids['Fz']
      for s in all_stats[g][c][f+'_MMN']:
        isp_data =  s[:, channel_id] * 1e6
        evkd_ax1.plot(time,isp_data,'k-', alpha=.15,
                     linewidth = 0.5)

      # 95% CI shades
      se_data = all_stats[g][c][f + '_MMN'][:, :, channel_id]
      stdev = np.std(se_data, 0)
      se = stdev/np.sqrt(se_data.shape[0])
      ci_upper = MMN + 1.96*1e6*se
      ci_lower = MMN - 1.96*1e6*se

      evkd_ax1.fill_between(time, ci_lower, ci_upper, color='k', alpha=.2)

      evkd_ax1.set_xlim([-0.25,0.4])
      evkd_ax1.set_ylim([-7,7])
      #evkd_ax1.legend(fontsize = 12, framealpha = 1, edgecolor = 'black',shadow = True)
  
  evkd_ax1.annotate(f, xy = (0.12,0.96 - spac*fidx),
                      xytext = (0.12,0.96- spac*fidx),
      xycoords = ('figure fraction','figure fraction'), textcoords='offset points',
      size=16, ha='left', va='top')
      
## Add legend
legend_elements = [mpl.lines.Line2D([0], [0], color='b', lw=4, label='Standard',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='r', lw=4, label='Deviant',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='k', lw=4, label='MMN',
                                    ls = '-')
                   ]

fig.legend(handles=legend_elements, loc=[0.35,0.005],ncol = 3, edgecolor = 'black',
           shadow = True,framealpha = 1)    

# print/save figure
# plt.tight_layout()
plt.savefig(wd + '/study2/results/first_figures.pdf') 

# %%
## Plot difference between conditions
fig = plt.figure(figsize = (10,5))
gs = gridspec.GridSpec(3,4,left=0.05, right=0.98,
                       top=0.93,bottom = 0.05,
                       wspace=0.35, hspace = 0.35,
                       width_ratios = [0.001,1,1,1],
                       height_ratios = [1,1,0.1])
channs = ['Fz']
features = ['intensity','timbre','location']
conds = ['optimal','hihat']
groups = ['controls','amusics']
spac = 0.315
channel_id = chann_ids['Fz']

for fidx,f in enumerate(features):
  for gidx,g in enumerate(groups):
    group_ax = plt.subplot(gs[gidx,0])
    group_ax.axis('off')
    group_ax.annotate(g, xy = (0.5,0.5),#xytext = (0.1,0.95),
                         # xycoords = ('figure fraction','figure fraction'), 
                          textcoords='offset points', size=13.5, ha='center',
                          va='center',rotation = 90)

    opt = grand_avg[g]['optimal'][f+'_MMN'].copy().pick_channels(channs).data.mean(0)*(1e6)
    hih = grand_avg[g]['hihat'][f + '_MMN'].copy().pick_channels(channs).data.mean(0)*(1e6)
    dif = grand_avg[g]['optimal'][f+'_MMN'].copy()
    dif = opt - hih
    time = grand_avg[g][c]['standard'].times
    evkd_ax1 = plt.subplot(gs[gidx,fidx+1])
    evkd_ax1.plot(time,opt,'b--',label = 'piano')
    evkd_ax1.plot(time,hih,'r--',label = 'hihat')
    evkd_ax1.plot(time,dif,'k-',label = 'difference')

    # individual subject difference traces
    # zip optimal and hihat conditions, iterate over both
    zip_conditions = zip(all_stats[g]['optimal'][f+'_MMN'], 
                         all_stats[g]['hihat'][f+'_MMN'])
    for s_opt, s_hh in zip_conditions:
        isp_opt = s_opt[:, channel_id] * 1e6
        isp_hh = s_hh[:, channel_id] * 1e6
        isp_data = isp_opt - isp_hh
        evkd_ax1.plot(time, isp_data, 'k-', alpha=.15,
                      linewidth=0.5)
    
    # 95% CI shades
    se_opt = all_stats[g]['optimal'][f + '_MMN'][:, :, channel_id]
    se_hh = all_stats[g]['hihat'][f + '_MMN'][:, :, channel_id]
    se_data = se_opt - se_hh
    stdev = np.std(se_data, 0)
    se = stdev/np.sqrt(se_data.shape[0])
    ci_upper = dif + 1.96*1e6*se
    ci_lower = dif - 1.96*1e6*se
    evkd_ax1.fill_between(time, ci_lower, ci_upper, color='k', alpha=.2)

    evkd_ax1.set_xlim([-0.25,0.4])
    evkd_ax1.set_ylim([-5.5,6])
      #evkd_ax1.legend(fontsize = 12, framealpha = 1, edgecolor = 'black',shadow = True)
  evkd_ax1.annotate(f, xy = (0.2 + spac*fidx,0.98),
                      xytext = (0.2+ spac*fidx,0.98),
      xycoords = ('figure fraction','figure fraction'), textcoords='offset points',
      size=16, ha='left', va='top')
      
## Add legend
legend_elements = [mpl.lines.Line2D([0], [0], color='b', lw=4, label='optimal',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='r', lw=4, label='hihat',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='k', lw=4, label='difference',
                                    ls = '-')
                   ]

fig.legend(handles=legend_elements, loc=[0.35,0.005],ncol = 3, edgecolor = 'black',
           shadow = True,framealpha = 1)  

# print/save figure
# plt.tight_layout()
plt.savefig(wd + '/study2/results/dif_figures.pdf') 

# %%
## Plot topography
times=[0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]

#f=plt.figure(figsize=(12,20))
#topogrid = gridspec.GridSpec(9,12,left=0.05, right=0.98,
                       #top=0.93,bottom = 0.05,
                       #wspace=0.35, hspace = 0.35,
                       #height_ratios = [1,1,1,1,1,1,1,1,1],
                       #width_ratios = [1,1,1,1,1,1,1,1,1,1,1,1]) 
                       
fig,axes=plt.subplots(nrows=12, ncols=9, figsize=(20,12), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 0.1]})

grand_avg['controls']['hihat']['location_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[0,0:8], colorbar=False)
grand_avg['controls']['optimal']['location_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[1,0:8], time_format='', colorbar=False)
grand_avg['amusics']['hihat']['location_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[2,0:8], time_format='', colorbar=False)
grand_avg['amusics']['optimal']['location_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[3,0:8], time_format='', colorbar=False)

grand_avg['controls']['hihat']['timbre_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[4,0:8], time_format='', colorbar=False)
grand_avg['controls']['optimal']['timbre_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[5,0:8], time_format='', colorbar=False)
grand_avg['amusics']['hihat']['timbre_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[6,0:8], time_format='', colorbar=False)
grand_avg['amusics']['optimal']['timbre_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[7,0:8], time_format='', colorbar=False)

grand_avg['controls']['hihat']['intensity_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[8,0:8], time_format='', colorbar=False)
grand_avg['controls']['optimal']['intensity_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[9,0:8], time_format='', colorbar=False)
grand_avg['amusics']['hihat']['intensity_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[10,0:8], time_format='', colorbar=False)
grand_avg['amusics']['optimal']['intensity_MMN'].plot_topomap(times=times,vmin=-6,vmax=6,axes=axes[11,0:8], time_format='', colorbar=False)

## Add colorbar
cmap = mpl.cm.RdBu_r
norm = mpl.colors.Normalize(vmin=-6, vmax=6)
cax = axes[0,8]
pos = cax.get_position()
new_pos = [pos.x0 + 0, pos.y0 + 0, pos.width*1, pos.height*1] 
cax.set_position(new_pos)
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical',
                                ticklocation = 'right')
cb1.ax.tick_params(labelsize=7,size = 0)
cb1.set_ticks([-6,0,6])

# print/save figure
plt.savefig(wd + '/study2/results/topoplot.pdf') 

