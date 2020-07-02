# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:22:18 2017

@author: david
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs
#plt.ion()

####
def runICA(subject,block,bad_channels,data_path,out_path):
    file_name = data_path + subject + '_' + block + '.bdf'    
    if not(os.path.exists(out_path)):
        os.mkdir(out_path)
        print('Directory created')         
    raw = mne.io.read_raw_bdf(file_name,preload=True)
    
    ## correct the cannel labels
    new_ch_names = {s: (s[2:] if s[0] == '1' else s) for s in raw.info['ch_names']}    
    raw.rename_channels(new_ch_names);
    
    ## set channel types and montage
    ch_types = {"vlEOG": "eog", "vhEOG": "eog","hlEOG": "eog", "hrEOG": "eog",
                'nose': "misc", 'rEAR': "misc",'EXG7': "misc", 'EXG8': "misc"}
    raw.set_channel_types(ch_types)
    montage = mne.channels.read_montage("biosemi64")
    raw.set_montage(montage)
    ## filter and picks
    raw.filter(1., 40., n_jobs=2)
    raw.info['bads'] =  bad_channels   
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                               stim=False,exclude ="bads")
 
    ## ICA                                                    
    n_components = 30  # if float, select n_components by explained variance of PCA
    method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
    decim = 3
    random_state = 23
    ica = ICA(n_components=n_components, method=method, random_state=random_state)
    print(ica)
    
    ica.fit(raw, picks=picks_eeg, decim=decim)
    print(ica)
     
    source_idx = range(0, ica.n_components_)
    ica_plot = ica.plot_components(source_idx, ch_type="eeg")
    
    eog_inds, eog_scores, eog_exclude = [], [], []
    picks_eeg2 = mne.pick_types(raw.info, meg=False, eeg=True, eog=True,
                               stim=False)
    eog_average = create_eog_epochs(raw, picks=picks_eeg2, ch_name = "vhEOG").average()
    eog_epochs = create_eog_epochs(raw, picks=picks_eeg2,ch_name = "vhEOG")  # get single EOG trials
    eog_inds, eog_scores = ica.find_bads_eog(eog_epochs,ch_name = "vhEOG")   # get single EOG trials
        
    if not eog_inds:
        if np.any(np.absolute(eog_scores)) > 0.3:
            eog_exclude=[np.absolute(eog_scores).argmax()]
            eog_inds=[np.absolute(eog_scores).argmax()]
            print("No EOG components above threshold were identified " + subject +
            " - selecting the component with the highest score under threshold above 0.3")
        elif not np.any(np.absolute(eog_scores)) > 0.3:
            eog_exclude=[]
            print("No EOG components above threshold were identified" + subject)
    elif eog_inds:
         eog_exclude += eog_inds
    
    ica.exclude += eog_exclude
    
    ica_plot.savefig(out_path + subject + '_' + block + '_comps_eog%s.pdf'% (str(eog_exclude)),format='pdf')
    
    ## Plotting ## 
    
    eog_source_plot = ica.plot_sources(eog_average, exclude=eog_inds)
    eog_score_plot = ica.plot_scores(eog_scores, exclude=eog_inds)
    eog_overlay_plot = ica.plot_overlay(eog_average, exclude=eog_inds) 
    eog_signal_plot = ica.plot_overlay(raw, exclude=eog_inds) 
       
    eog_source_plot.savefig(out_path + subject + '_' + block + '_eog_sources.pdf',format='pdf')
    eog_score_plot.savefig(out_path + subject + '_' + block + '_eog_scores.pdf',format='pdf')
    eog_overlay_plot.savefig(out_path + subject + '_' + block + '_eog_overlay.pdf',format='pdf')
    eog_signal_plot.savefig(out_path + subject + '_' + block + '_eog_signal.pdf',format='pdf')
    
    plt.close('all')
    
    ## Save ICA
#    raw2 = mne.io.read_raw_edf(file_name,preload=True)
#    new_ch_names = {s: (s[2:] if s[0] == '1' else s) for s in raw2.info['ch_names']}    
#    raw2.rename_channels(new_ch_names);
#    raw2.set_channel_types(ch_types)
#    raw2.set_montage(montage)
#    raw2 = ica.apply(raw2)
#    raw2.info['bads'] = bad_channels
#    raw2.interpolate_bads(reset_bads=False) # interpolate bad channels
#    raw2.save(out_path + subject + '_' + block + '_ica-raw.fif', overwrite=True)
    ica.save(out_path + subject + '_' +  block + '-ica.fif')


