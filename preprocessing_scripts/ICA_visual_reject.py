# -*- coding: utf-8 -*

import matplotlib.pyplot as plt
import pandas as pnd
import mne
from mne.preprocessing import create_eog_epochs
plt.ion()

wd = 'C:/Users/au571303/Documents/projects/amusia_study'
   
subject = 27
block_names =  ['optimal','alberti','melody','familiar',
                'unfamiliar','hihat','piazzolla','hungarian']

idx = [0,1,2,3,4,5,6,7] # block to process
#idx = [2]

blocks = [block_names[x] for x in idx]

raw_path     = wd + "/data/raw/"
ica_path     = wd + "/data/ICA/"

subjects_info = pnd.read_csv(wd +  '/misc/subjects_info.csv', sep = ';')
if not (subjects_info.bads[subjects_info.subject == subject] == 'none').bool():
    bad_channels = list(subjects_info.bads[subjects_info.subject == subject])[0].split(",");
else:
    bad_channels = []  

for block in blocks:
    print(block)
    sub = str(subject)
    icacomps = mne.preprocessing.read_ica(ica_path + sub + '_' + block + '-ica.fif')
    if icacomps.exclude:
        print('##################')
        print('Pre-selected comps: '+str(icacomps.exclude))
        print('##################')
        icacomps.excludeold=icacomps.exclude
        icacomps.exclude=[]
    if not icacomps.exclude:
        print('Old components copied. Exclude field cleared')
    raw_fname = raw_path + sub + '_' + block + '.bdf'

    raw = mne.io.read_raw_bdf(raw_fname,preload=True)
    new_ch_names = {s: (s[2:] if s[0] == '1' else s) for s in raw.info['ch_names']}    
    raw.rename_channels(new_ch_names);
    ch_types = {"vlEOG": "eog", "vhEOG": "eog","hlEOG": "eog", "hrEOG": "eog",
                'nose': "misc", 'rEAR': "misc",'EXG7': "misc", 'EXG8': "misc"}
    raw.set_channel_types(ch_types)
        ## correct the cannel labels

    montage = mne.channels.read_montage("biosemi64")
    raw.set_montage(montage)
    
    raw.filter(1., 40.)
    raw.info['bads'] = bad_channels
    raw.interpolate_bads(reset_bads=True) 
    eog_picks = mne.pick_types(raw.info, meg=False, eeg=False, ecg=False, eog=True,
                   stim=False, exclude='bads')[0]
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, ecg=False,
                       stim=False, exclude='bads')               
                   
    eog_evoked = create_eog_epochs(raw, tmin=-.5, tmax=.5,picks=eeg_picks,
                           ch_name='vhEOG', verbose=False).average()

    # ica topos
    source_idx = range(0, icacomps.n_components_)
    ica_plot = icacomps.plot_components(source_idx, ch_type="eeg") 
    plt.waitforbuttonpress(1)
    
    title = 'Sources related to %s artifacts (red)'
    
    #ask for comps ECG
    prompt = '> '
    eog_done = 'N'
        
    while eog_done.strip() != 'Y' and eog_done.strip() != 'y':
        eog_source_idx = []        
        print('##################')
        print('Pre-selected EOG comps : '+str(icacomps.excludeold))
        print('##################')        
        print('What components should be rejected as EOG comps?')
        print('If more than one, list them each separated by a comma and a space')
        print('And if none, just hit ENTER')
        try:
            eog_source_idx = list(map(int,input(prompt).split(',')))
        except ValueError:
            eog_source_idx = []
            print('##################')
            print('Exiting EOG - No components selected')
            break
               
        print(eog_source_idx)
        icacomps2 = icacomps
        icacomps2.exclude = eog_source_idx
        if eog_source_idx: 
            print(eog_source_idx)
            source_plot_eog = icacomps2.plot_sources(eog_evoked)
            plt.waitforbuttonpress(1)
            clean_plot_eog=icacomps2.plot_overlay(eog_evoked, exclude=eog_source_idx)
            plt.waitforbuttonpress(1)
            print('##################')
            print('Clean enough?[Y/N]: ')
            print('')
            print('To terminate without selecting any components, type "N" now')
            print('and then don''t select any components pressing ENTER')
            eog_done = input(prompt)
            plt.close(source_plot_eog)
            plt.close(clean_plot_eog)
    eog_exclude = eog_source_idx
    
    if eog_source_idx:
        icacomps.exclude += eog_source_idx
        source_plot_eog.savefig(ica_path + 'vis_' + sub + '_' + block + '_eog_source.pdf', format = 'pdf')
#        plt.waitforbuttonpress(1)
        clean_plot_eog.savefig(ica_path + 'vis_' + sub + '_' + block +'_eog_clean.pdf', format = 'pdf')
#        plt.waitforbuttonpress(1)
        plt.close(source_plot_eog)
        plt.close(clean_plot_eog) 
    else:
        print('*** No EOG components rejected...')
    
    print('############')    
    print('*** Excluding following components: ', icacomps.exclude)
    print('')
    ica_plot.savefig(ica_path + 'vis_' + sub + '_' + block + '_comps_eog%s.pdf'% (str(eog_exclude)),format='pdf')
    plt.close('all')
    icacomps.save(ica_path + sub + '_' + block + '-ica.fif')
    
    # raw2 = mne.io.read_raw_edf(raw_fname,preload=True)
    # new_ch_names = {s: (s[2:] if s[0] == '1' else s) for s in raw2.info['ch_names']}    
    # raw2.rename_channels(new_ch_names);
    # ch_types = {"vlEOG": "eog", "vhEOG": "eog","hlEOG": "eog", "hrEOG": "eog",
    #             'nose': "misc", 'rEAR': "misc",'EXG7': "misc", 'EXG8': "misc"}
    # raw2.set_channel_types(ch_types)
    # raw2.set_montage(montage)
    
    # raw2 = icacomps.apply(raw2)
    # raw2.info['bads'] = bad_channels
    # raw2.interpolate_bads(reset_bads=True)
    # raw2.save(wd + '/ICA_export/' + sub + '_' + block + '_ica-raw.fif', overwrite=True,verbose=False)



