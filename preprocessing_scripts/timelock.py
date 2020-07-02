# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:30:09 2018

@author: au571303
"""
import os
import os.path as op
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
#plt.ion()

def timelock(subject,condition,std_range,data_dir,
             ica_dir,fig_dir,out_dir,bads,tmin,tmax,chan_plot):
    
    if not op.exists(fig_dir):
        os.makedirs(fig_dir)
    if not op.exists(out_dir):
        os.makedirs(out_dir)
     
    ## Start script ##
    raw_fname = op.join(data_dir, '{:d}_{:s}.bdf'.format(subject,condition))
    ica_fname = op.join(ica_dir, '{:d}_{:s}-ica.fif'.format(subject,condition))
    raw = mne.io.read_raw_bdf(raw_fname, preload = True)
    
    new_ch_names = {s: (s[2:] if s[0] == '1' else s) for s in raw.info['ch_names']}    
    raw.rename_channels(new_ch_names);
    ch_types = {"vlEOG": "eog", "vhEOG": "eog","hlEOG": "eog", "hrEOG": "eog",
                'nose': "misc", 'rEAR': "misc",'EXG7': "misc", 'EXG8': "misc"}
    raw.set_channel_types(ch_types)
        ## correct the cannel labels

    montage = mne.channels.read_montage("biosemi64")
    raw.set_montage(montage)
    raw.info['bads'] =  bads   
    raw.interpolate_bads(reset_bads=True) 
    ica = mne.preprocessing.read_ica(ica_fname)
    raw = ica.apply(raw)
#    raw.set_eeg_reference(ref_channels = ['nose'])
    raw.set_eeg_reference(ref_channels = ['P9','P10'])
    raw.notch_filter(50, filter_length='auto',phase='zero')
    raw.filter(0.1, 40, fir_design='firwin')
    
    events	 = mne.find_events(raw, shortest_event = 1)
    events = events[np.append(np.diff(events[:,0]) > 2,True)] # delete unwanted triggers
    events[(events[:,2] <= std_range),2] = 500 # create code for standards
    events = events[(events[:,2] > 240),:] # clean the code list
    dev_mask = np.where(events[:,2] != 500)[0] + 1  # find index of standards that follow a deviant
    dev_mask = dev_mask[dev_mask <= len(events)] # restriction for last tone
    events = np.delete(events,dev_mask,axis = 0) # don't include standards following a deviant
    events[events[:,2] != 500,0] = events[events[:,2] != 500,0] - 49 # correct trigger onset
    
#    raw.plot(n_channels=10, block=True, events = events)
    
   # tmin, tmax = -0.25, 0.4
    event_id = {'standard': 500, 'pitch': 241, 'intensity': 242,
                'timbre': 243, 'location': 244}#, 'standard': 500}#, 'Rhythm_std': 235}
    
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)
#    baseline = (-0.11, -0.06)
    baseline = (-0.1,0)
    reject =  dict(eeg=150e-6)#, eog =150e-6)
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                        tmax=tmax, picks=picks, baseline = baseline, reject = reject)
    rhythm_id = {'rhythm': 245}
    rhythm_epochs = mne.Epochs(raw, events=events, event_id=rhythm_id, tmin=tmin,
                        tmax=tmax, picks=picks, baseline = (-0.1,0), reject = reject)
    
    rhythm_id2 = {'rhythm2': 245}
    rhythm_events2 = events.copy(); rhythm_events2[:,0] = rhythm_events2[:,0] + 60
    rhythm_epochs2 = mne.Epochs(raw, events=rhythm_events2, event_id=rhythm_id2, tmin=tmin,
                        tmax=tmax, picks=picks, baseline = (-0.1,0), reject = reject)
    
    standard_id2 = {'standard_rhy': 500}
    standard_epochs2 = mne.Epochs(raw, events= events, event_id=standard_id2, tmin=tmin,
                        tmax=tmax, picks=picks, baseline = (-0.140,-0.04), reject = reject)
    
    all_evokeds = dict((cond,epochs[cond].average()) for cond in sorted(event_id.keys()))
    all_evokeds['rhythm'] = rhythm_epochs.average()
    all_evokeds['rhythm2'] = rhythm_epochs2.average()
    all_evokeds['standard_rhy'] = standard_epochs2.average()
    
    features = ['pitch', 'intensity','timbre','location','rhythm']
    
    ch_names = all_evokeds['standard'].info['ch_names']
    
    channs = mne.pick_channels(ch_names,include = chan_plot)#['F1','Fz','F2','FC1','FCZ','FC2','C1','Cz','C2'])
    fig, ax = plt.subplots(5, figsize=(10, 15))
    
    n = -1
    for feat in features:
        n = n + 1
        ERPs = {}
        ERPs['deviant'] = all_evokeds[feat]
        if feat == 'rhythm':
           ERPs['standard'] = all_evokeds['standard_rhy']
        else:
           ERPs['standard'] = all_evokeds['standard']
           
        ERPs['difference'] = mne.combine_evoked([ERPs['deviant'],ERPs['standard']],weights=[1,-1])
        all_evokeds[feat + '_MMN'] = ERPs['difference']
        all_evokeds[feat + '_MMN'].comment = feat + '_MMN'
        if n == 0: show_legend = 3
        else: show_legend = False
        vlines = [-0.25,0,0.25]
    #    if n == 4: vlines = [-0.25,-0.04,0,0.25,0.5,0.75,1,1.25]        
        mne.viz.plot_compare_evokeds(axes = ax[n],evokeds = ERPs, colors = ['red','blue','black'],
                                     combine = 'mean',ylim = dict(eeg = [-8,8]),title = feat,
                                     linestyles = dict(difference='--', deviant='-',standard = '-'),
                                     picks = channs, show_legend = show_legend, show = False,
                                     styles = {"standard": {"linewidth": 2},"deviant": {"linewidth": 2}},
                                     vlines = vlines)
        if n != 4: ax[n].set_xlabel('')  
        if n == 0: ax[n].legend(loc=3, prop={'size': 10})
        
    plt.tight_layout()
    plt.savefig(fig_dir + '{:d}_{:s}.pdf'.format(subject,condition))
    
    plt.close('all')
    evokeds_list = []
    for key, value in all_evokeds.items():
        evokeds_list.append(value)
        
    mne.write_evokeds((out_dir + str(subject) + '_' + condition + '_timelocked-ave.fif'),evokeds_list)
    
    fig, ax = plt.subplots(5,11, figsize=(20, 15))
    
    n = -1
    for feat in features:
        n = n + 1
        ERPs = {}
        ERPs['deviant'] = all_evokeds[feat]
        if feat == 'rhythm':
           ERPs['standard'] = all_evokeds['standard_rhy']
        else:
           ERPs['standard'] = all_evokeds['standard']
        ERPs['difference'] = mne.combine_evoked([ERPs['deviant'],ERPs['standard']],weights=[1,-1])
        times = [0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.250,0.275,0.3]
        ERPs['difference'].plot_topomap(axes = ax[n,],times = times,
                                        vmin = -3,vmax = 3)
        ax[n,0].set_title(feat)
        if n != 0:
            for t in range(1,len(times)):
                ax[n,t].set_title('')
             
    plt.savefig(fig_dir + '{:d}_{:s}_topo.pdf'.format(subject,condition)) 
    plt.close('all')
    
    all_evokeds['standard'].plot_joint()
    plt.savefig(fig_dir + '{:d}_{:s}_standards_joint.pdf'.format(subject,condition))
    ch_names = all_evokeds['standard'].info['ch_names']
    channs = mne.pick_channels(ch_names,include = chan_plot)
    
    mne.viz.plot_compare_evokeds(evokeds = all_evokeds['standard'],ylim = dict(eeg = [-8,8]),
                                 title = 'standards',picks = channs, combine = 'mean')
    
    plt.savefig(fig_dir + '{:d}_{:s}_standards_Fz.pdf'.format(subject,condition))
    plt.close('all')
    
    ########### Rhythm issues ###############
    ERPs_beat={}
    ERPs_beat['standard'] = all_evokeds['standard']
    ERPs_beat['deviant'] = all_evokeds['rhythm2'] 
    ERPs_beat['difference'] = mne.combine_evoked([ERPs_beat['deviant'],
                                                  ERPs_beat['standard']],weights=[1,-1])
    
    ERPs_no_beat={}
    ERPs_no_beat['standard'] = all_evokeds['standard_rhy']
    ERPs_no_beat['deviant'] = all_evokeds['rhythm']  
    ERPs_no_beat['difference'] = mne.combine_evoked([ERPs_no_beat['deviant'],
                                                     ERPs_no_beat['standard']],weights=[1,-1])
    
    ch_names = all_evokeds['standard'].info['ch_names']
    channs = mne.pick_channels(ch_names,include = chan_plot)
    #['F1','Fz','F2','FC1','FCZ','FC2','C1','Cz','C2'])
    fig, ax = plt.subplots(2,1, figsize=(20, 15))
    
    vlines = [-0.25,0,0.25,0.4]     
    
    mne.viz.plot_compare_evokeds(axes = ax[0],evokeds = ERPs_no_beat, colors = ['blue','red','black'],
                                 combine = 'mean',ylim = dict(eeg = [-8,8]),title = 'timelocked to deviant onset',
                                 linestyles = dict(difference='--', deviant='-',standard = '-'),
                                 picks = channs, show_legend = 3,show = False,
                                 styles = {"standard": {"linewidth": 2},"deviant": {"linewidth": 2}},
                                 vlines = vlines)
    
    vlines = [-0.25,-0.06,0,0.25,0.4]       
    
    mne.viz.plot_compare_evokeds(axes = ax[1],evokeds = ERPs_beat, colors = ['blue','red','black'],
                                 combine = 'mean',ylim = dict(eeg = [-8,8]),title = 'timelocked to beat',
                                 linestyles = dict(difference='--', deviant='-',standard = '-'),
                                 picks = channs, show_legend = False,show = False,
                                 styles = {"standard": {"linewidth": 2},"deviant": {"linewidth": 2}},
                                 vlines = vlines)
       
    plt.tight_layout()
    plt.savefig(fig_dir + '{:d}_{:s}_rhythm_issues.pdf'.format(subject,condition))
    plt.close('all')