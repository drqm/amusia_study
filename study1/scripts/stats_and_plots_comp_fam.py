# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:09:15 2018

@author: au571303
"""
#matplotlib.use('Qt4Agg')
#import importlib as il
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import pandas as pnd
import mne
import numpy  as np
import pickle
from scipy import stats as stats
import os

wd = 'C:/Users/au571303/Documents/projects/amusia_study'
os.chdir(wd + '/study1/scripts')
import make_plots as mp
os.chdir(wd)

plt.ion()

###### Set relevant varaibles ######################
groups = ['controls','amusics']
#subjects = [2,3,10]#[1,4,5,6,7,8,9,11]#list(range(1,8)) [2,3]
data_dir = wd + '/data/timelock/data/'
out_dir = wd + '/study1/results/'
conditions = ['optimal','alberti','melody','familiar','unfamiliar']
do_stats = 0 # 1 will run stats, no otherwise
extract_MA1 = 0 # 1 will extract and save mean amplitudes
extract_MA2 = 0 # 1 will extract MA per row of electrodes
extract_MA3 = 0 # 1 will extract MA around grand-mean peaks

cond_names = ['low complexity','int. complexity','high complexity',
              'familiar','unfamiliar']

exclusions = {'optimal': [0],
              'alberti': [0],
              'melody': [0],
              'familiar': [24,31],
              'unfamiliar': [24,31]}

subjects_info = pnd.read_csv(wd +  '/misc/subjects_info.csv', sep = ';')
subjects = subjects_info[['subject','group']]
all_groups = {'controls': {}, 'amusics': {}, 'all': {}}
all_stats = {'controls': {}, 'amusics': {}, 'all': {}}
sub_mapping = {'controls': {}, 'amusics': {}, 'all': {}}

################# Load the data in a single dictionary "all_groups"##########
group_counts = {'amusic': 0, 'control': 0, 'all': 0}
for idx, row in subjects.iterrows():
    print(idx, 'loading subject:',row['subject'], '   group:', row['group'])
    sub, group = row['subject'], row['group']
    group_counts[group] = group_counts[group] + 1
    group_counts['all'] = group_counts['all'] + 1
    
    for cond in conditions:
        if np.logical_not(np.isin(row['subject'],exclusions[cond])):
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
                                                 
               # organize the data for the cluster permutations
               # we need a (subjects x time x chanels) 3D array
               
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
               print(evkd)
   
########################### Grand average the data #########################
# why does the grand_avg scale the data? factor seems to be 17
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

#####
            
#ratio = np.divide(np.transpose(grand_avg['controls']['optimal']['intensity_MMN'].data,(1,0)),
#        np.mean(all_stats['controls']['optimal']['pitch_MMN'],0))            
########################## Extract Mean Amplitudes #########################

if extract_MA1:
    # First identify channels with largest P50.
    
    ## average across all subjects:
               
    # grand_stack = np.dstack((grand_avg['all']['optimal']['standard'].data,
    #                         grand_avg['all']['alberti']['standard'].data,
    #                         grand_avg['all']['melody']['standard'].data,
    #                         grand_avg['all']['familiar']['standard'].data,
    #                         grand_avg['all']['unfamiliar']['standard'].data))
    
    # great_avg = np.mean(grand_stack,2)
    
    # # instpect the great average:
    # great_evkd = grand_avg['controls']['optimal']['standard'].copy()
    # great_evkd.data = great_avg
    # great_evkd.plot_joint() 
    
    # ## find channels with max P50
    # times = great_evkd.times
    # times_P50 = np.where((times >= 0.036) & (times <= 0.096))[0] #select P50 TW  
    # p50_max = np.amax(great_avg[:,times_P50],1) # find max value
    # sort_idx = np.argsort(p50_max) # sort channels according to max value
    
    # last_idx = sort_idx[-11:-1] # select the 10 channels with the largest P50
    # chans = [great_evkd.info['chs'][i]['ch_name'] for i in last_idx]
    
    
    # # inspect the channels:
    # great_evkd.plot_joint(picks = chans) 
    # great_evkd.plot_joint()
    
    # # pick channels for mean averages:
    
    chan_sel = ['Fz','F1','F2','F3','F4','FCz','FC1','FC2','FC3','FC4']   
    times = grand_avg['controls']['optimal']['pitch_MMN'].times
    
    features = ['pitch','intensity','timbre','location','rhythm']
    conditions = ['optimal','alberti','melody','familiar','unfamiliar']
    MA_data = pnd.DataFrame() # pandas df to store the averages
    
    for idx, row in subjects_info.iterrows():
        sgroup = row['group'] + 's' # fetch subject's group
        scode = row['subject'] # fetch subject's code
        mbea = row['Mean MBEA']
        mbea_pitch = row['Mean MBEA Pitch']
        PDT = row['PDT']
        age = row['age']
        for c in conditions:
            for f in features:
                t1 = 0.07
                if f == 'pitch': 
                   t2 = 0.3 # we select a slightly longer TW for pitch
                else:
                   t2 = 0.25                  
                # map s code to s index in the evoked arrays
                smap = sub_mapping[sgroup][c][f + '_MMN'] 
                sidx = np.where(smap[:,0] == scode)[0]
                # only if the subject's code was found, extract MA:
                if sidx.size > 0:
                   cur_evkd = all_groups[sgroup][c][f + '_MMN'][sidx[0]].copy()
                   cur_evkd.pick_channels(chan_sel)
                   cur_data = np.mean(cur_evkd.data,0)
                   peak_idx = ((cur_data[1:-1] - cur_data[2:] < 0) &
                              (cur_data[1:-1] - cur_data[0:-2] < 0) &
                              (times[1:-1] >= t1) & (times[1:-1] <= t2))
                   cur_data = cur_data[1:-1]
                   peak_idx2 = np.where((peak_idx == True) &
                                        (cur_data == np.amin(cur_data[peak_idx])))
                   latency = times[1:-1][peak_idx2]
                   tw = np.where((times[1:-1] >= latency - 0.025) & 
                                 (times[1:-1] <= latency + 0.025))
                   MA = np.mean(cur_data[tw])
                   latency = np.round(latency*1000)[0]
                   currow = pnd.DataFrame({"subject": [scode], 
                                               "group": sgroup,
                                               "age": age,
                                               "MBEA": mbea,
                                               "MBEA_pitch": mbea_pitch,
                                               "PDT": PDT,
                                               "condition": c,
                                               "feature": f,
                                               "latency": [latency],
                                               "amplitude": [MA]}) 
                   if MA_data.empty:
                      MA_data = currow
                   else:
                      MA_data = MA_data.append(currow)
    
    # and save to csv:
    MA_data.to_csv(wd + '/data/MA/MA.csv', index = None, header = True)
    
##############################################################################
     
################ Extract Mean Amplitudes per electrode row ###################

if extract_MA2:
    # First identify time windows around the peak for each feature and condition.
    
    features = ['pitch','intensity','timbre','location']#,'rhythm']
    comparisons = ['complexity','familiarity']

    ## pick channels for mean averages:
    
    chan_sel = ['AFz','AF1','AF2','AF3','AF4','Fz','F1','F2','F3','F4',
                'FCz','FC1','FC2','FC3','FC4','Cz','C1','C2','C3','C4',
                'CPz','CP1','CP2','CP3','CP4']
    erow_sel = [['AFz','AF1','AF2','AF3','AF4'],['Fz','F1','F2','F3','F4'],
               ['FCz','FC1','FC2','FC3','FC4'],['Cz','C1','C2','C3','C4'],
               ['CPz','CP1','CP2','CP3','CP4']]
    erow_id = ['anterior','frontal','frontocentral','central','centroparietal']
    
    ## Now let's extract mean amplitudes
    
    times = grand_avg['controls']['optimal']['pitch_MMN'].times
    
    features = ['pitch','intensity','timbre','location','rhythm']
    conditions = ['optimal','alberti','melody','familiar','unfamiliar']
    MA_data = pnd.DataFrame() # pandas df to store the averages
    
    for idx, row in subjects_info.iterrows():
        sgroup = row['group'] + 's' # fetch subject's group
        scode = row['subject'] # fetch subject's code
        mbea = row['Mean MBEA']
        mbea_pitch = row['Mean MBEA Pitch']
        PDT = row['PDT']
        age = row['age']
        for c in conditions:
            for f in features:
                t1 = 0.07
                if f == 'pitch': 
                   t2 = 0.3 # we select a slightly longer TW for pitch
                else:
                   t2 = 0.25                  
                # map s code to s index in the evoked arrays
                smap = sub_mapping[sgroup][c][f + '_MMN'] 
                sidx = np.where(smap[:,0] == scode)[0]
                # only if the subject's code was found, extract MA:
                if sidx.size > 0:
                   cur_evkd = all_groups[sgroup][c][f + '_MMN'][sidx[0]].copy()
                   cur_evkd.pick_channels(chan_sel)
                   cur_data = np.mean(cur_evkd.data,0)
                   peak_idx = ((cur_data[1:-1] - cur_data[2:] < 0) &
                              (cur_data[1:-1] - cur_data[0:-2] < 0) &
                              (times[1:-1] >= t1) & (times[1:-1] <= t2))
                   cur_data = cur_data[1:-1]
                   peak_idx2 = np.where((peak_idx == True) &
                                        (cur_data == np.amin(cur_data[peak_idx])))
                   latency = times[1:-1][peak_idx2]
                   tw = np.where((times[1:-1] >= latency - 0.025) & 
                                 (times[1:-1] <= latency + 0.025))
                   latency = np.round(latency*1000)
                   for eidx in range(len(erow_id)):
                       cur_eevkd = cur_evkd.copy().pick_channels(erow_sel[eidx])
                       cur_edata = np.mean(cur_eevkd.data,0)
                       cur_edata = cur_edata[1:-1]
                       MA = np.mean(cur_edata[tw])
                       currow = pnd.DataFrame({"subject": [scode], 
                                                   "group": sgroup,
                                                   "age": age,
                                                   "MBEA": mbea,
                                                   "MBEA_pitch": mbea_pitch,
                                                   "PDT": PDT,
                                                   "condition": c,
                                                   "feature": f,
                                                   "latency": latency,
                                                   "erow": erow_id[eidx],
                                                   "amplitude": [MA]}) 
                       if MA_data.empty:
                          MA_data = currow
                       else:
                          MA_data = MA_data.append(currow)
    
    # and save to csv:
    MA_data.to_csv(wd + '/data/MA/MA_erows.csv', index = None, header = True)
    
##############################################################################

############# Extract Mean Amplitudes around grand_avg peak ##################

if extract_MA3:
    
    chan_sel = ['Fz','F1','F2','F3','F4','FCz','FC1','FC2','FC3','FC4']   
    times = grand_avg['controls']['optimal']['pitch_MMN'].times    
    features = ['pitch','intensity','timbre','location','rhythm']
    conditions = ['optimal','alberti','melody','familiar','unfamiliar']
    MA_data = pnd.DataFrame() # pandas df to store the averages
    
    for idx, row in subjects_info.iterrows():
        sgroup = row['group'] + 's' # fetch subject's group
        scode = row['subject'] # fetch subject's code
        mbea = row['Mean MBEA']
        mbea_pitch = row['Mean MBEA Pitch']
        PDT = row['PDT']
        age = row['age']
        for c in conditions:
            for f in features:
                t1 = 0.07
                if f == 'pitch': 
                   t2 = 0.3 # we select a slightly longer TW for pitch
                else:
                   t2 = 0.25                  
                # map s code to s index in the evoked arrays
                smap = sub_mapping[sgroup][c][f + '_MMN'] 
                sidx = np.where(smap[:,0] == scode)[0]
                # only if the subject's code was found, extract MA:
                if sidx.size > 0:
                    # current individual data:
                   cur_evkd = all_groups[sgroup][c][f + '_MMN'][sidx[0]].copy()
                   cur_evkd.pick_channels(chan_sel)
                   cur_data = np.mean(cur_evkd.data,0)
                   
                   # current grand average data:
                   great_data = grand_avg['all'][c][f + '_MMN']
                   great_data.pick_channels(chan_sel)
                   cur_great = np.mean(great_data.data,0)
                   
                   #extract peak latency for individual subject:
                   peak_idx = ((cur_data[1:-1] - cur_data[2:] < 0) &
                              (cur_data[1:-1] - cur_data[0:-2] < 0) &
                              (times[1:-1] >= t1) & (times[1:-1] <= t2))
                   cur_data = cur_data[1:-1]
                   peak_idx2 = np.where((peak_idx == True) &
                                        (cur_data == np.amin(cur_data[peak_idx])))
                   latency = times[1:-1][peak_idx2]
                   
                   #extract peak latency for grand average:
                       
                   peak_idx3 = ((cur_great[1:-1] - cur_great[2:] < 0) &
                              (cur_great[1:-1] - cur_great[0:-2] < 0) &
                              (times[1:-1] >= t1) & (times[1:-1] <= t2))
                   cur_great = cur_great[1:-1]
                   peak_idx4 = np.where((peak_idx3 == True) &
                                        (cur_great == np.amin(cur_great[peak_idx3])))
                   latency2 = times[1:-1][peak_idx4]
                   
                   # Average amplitudes around found time window:
                   tw = np.where((times[1:-1] >= latency2 - 0.025) & 
                                 (times[1:-1] <= latency2 + 0.025))
                   MA = np.mean(cur_data[tw])
                   latency = np.round(latency*1000)[0]
                   latency2 = np.round(latency2*1000)[0]
                   currow = pnd.DataFrame({"subject": [scode], 
                                               "group": sgroup,
                                               "age": age,
                                               "MBEA": mbea,
                                               "MBEA_pitch": mbea_pitch,
                                               "PDT": PDT,
                                               "condition": c,
                                               "feature": f,
                                               "latency": [latency],
                                               "latency2": [latency2],
                                               "amplitude": [MA]}) 
                   if MA_data.empty:
                      MA_data = currow
                   else:
                      MA_data = MA_data.append(currow)
    
    # and save to csv:
    MA_data.to_csv(wd + '/data/MA/MA_grand_window.csv', index = None, header = True)
    
##############################################################################

############################### STATS ########################################
      
#tmin_stats = 0.1
#tmax_stats = 0.3 
times = grand_avg['controls']['optimal']['pitch_MMN'].times
time_stats = np.where((times >= 0) & (times <= 0.3))[0]  
nperm = 10000

# compute neighbor channels: 
info_ch = grand_avg['controls']['optimal']['pitch_MMN'].info
con = mne.channels.find_ch_connectivity(info_ch, ch_type = 'eeg')
#con2 = mne.channels.read_ch_connectivity('biosemi64')
##############################################################################

############################ Check MMN #######################################
if do_stats:
    features = ['pitch','intensity','timbre','location','rhythm']
    
    p_threshold = 0.001
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., 17 - 1)
    MMN_check = {}
    for f in features:
        MMN_check[f + '_MMN'] = {}
        for g in groups:
            MMN_check[f + '_MMN'][g] = {}
            for c in conditions:
                cur_data = all_stats[g][c][f +  '_MMN'][:,time_stats,:]
                MMN_check[f + '_MMN'][g][c] = mne.stats.spatio_temporal_cluster_1samp_test(
                                                            cur_data,
                                                            connectivity = con[0],
                                                            out_type='mask',
                                                            n_permutations = nperm)
    
    with open(out_dir + 'MMN_check.p', 'wb') as fp:
        pickle.dump(MMN_check, fp, protocol=pickle.HIGHEST_PROTOCOL)
              
    ########################## F TEST COMPLEXITY: ############################
    
    time_stats_comp = np.where((times >= 0) & (times <= 0.3))[0]  
    factor_levels = [3]
    effects = 'A'
    pthresh = 0.05
    n_subjects = 17
    f_thresh = mne.stats.f_threshold_mway_rm(n_subjects, factor_levels, effects, pthresh)
    
    def stat_fun(*args):
        # get f-values only.
        return mne.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                         effects=effects, return_pvals=False)[0]
    
    def stat_fun_t(*args):
        # get t-values only.
        return stats.ttest_ind(args[0],args[1],equal_var=False)[0]
    
    features = ['pitch','intensity','timbre','location','rhythm'] 
      
    F_diff = {}
    out_stats= {}
    
    for f in features:
        F_diff[f] = {}
        out_stats[f] = {}
        out_stats[f]['interaction'] = {}
        out_stats[f]['main'] = {}
        
        F_main = [all_stats['all']['optimal'][f + '_MMN'][:,time_stats_comp,:],
                  all_stats['all']['alberti'][f + '_MMN'][:,time_stats_comp,:],
                  all_stats['all']['melody'][f + '_MMN'][:,time_stats_comp,:]]
        out_stats[f]['main'] = mne.stats.spatio_temporal_cluster_test(F_main, 
                                                            threshold = f_thresh,
                                                            stat_fun = stat_fun, 
                                                            n_permutations = nperm,
                                                            connectivity = con[0],
                                                            out_type='mask')    
        for g in groups:
            F_diff[f][g] = {}
            out_stats[f][g] = {}
            F_conds = [all_stats[g]['optimal'][f + '_MMN'][:,time_stats_comp,:],
                      all_stats[g]['alberti'][f + '_MMN'][:,time_stats_comp,:],
                      all_stats[g]['melody'][f + '_MMN'][:,time_stats_comp,:]]
                  
            F_diff[f][g]['ao'] = F_conds[1] - F_conds[0]  
            F_diff[f][g]['mo'] = F_conds[2] - F_conds[0]  
            F_diff[f][g]['ma'] = F_conds[2] - F_conds[1]  
            
            out_stats[f][g]['Ftest'] = mne.stats.spatio_temporal_cluster_test(F_conds, 
                                                                threshold = f_thresh,
                                                                stat_fun = stat_fun, 
                                                                n_permutations = nperm,
                                                                connectivity = con[0],
                                                                out_type='mask')
            out_stats[f][g]['pw_ao'] = mne.stats.spatio_temporal_cluster_1samp_test(F_diff[f][g]['ao'],
                                                                                    connectivity = con[0],
                                                                                    n_permutations = nperm,
                                                                                    out_type='mask')
            out_stats[f][g]['pw_mo'] = mne.stats.spatio_temporal_cluster_1samp_test(F_diff[f][g]['mo'],
                                                                                    connectivity = con[0],
                                                                                    n_permutations = nperm,
                                                                                    out_type='mask')
            out_stats[f][g]['pw_ma'] = mne.stats.spatio_temporal_cluster_1samp_test(F_diff[f][g]['ma'],
                                                                                    connectivity = con[0],
                                                                                    n_permutations = nperm,
                                                                                    out_type='mask')
       
        out_stats[f]['interaction']['ao'] = mne.stats.spatio_temporal_cluster_test([F_diff[f]['amusics']['ao'],
                                                                                    F_diff[f]['controls']['ao']],
                                                                                    connectivity = con[0],
                                                                                    stat_fun = stat_fun_t,
                                                                                    threshold = 2,
                                                                                    n_permutations = nperm,
                                                                                    out_type = 'mask')
        out_stats[f]['interaction']['mo'] = mne.stats.spatio_temporal_cluster_test([F_diff[f]['amusics']['mo'],
                                                                                    F_diff[f]['controls']['mo']],
                                                      connectivity = con[0],
                                                      stat_fun = stat_fun_t,
                                                      threshold = 2,
                                                      n_permutations = nperm,
                                                      out_type = 'mask')  
        out_stats[f]['interaction']['ma'] = mne.stats.spatio_temporal_cluster_test([F_diff[f]['amusics']['ma'],
                                                                                    F_diff[f]['controls']['ma']],
                                                      connectivity = con[0],
                                                      stat_fun = stat_fun_t,
                                                      threshold = 2,
                                                      n_permutations = nperm,
                                                      out_type = 'mask')  
            
    
    with open(out_dir + 'complexity_stats.p', 'wb') as fp:
        pickle.dump(out_stats, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
##############################################################################
    
######################## T-tests familiarity #################################
    
    time_stats_fam = np.where((times >= 0) & (times <= 0.3))[0]  
    # prepare data:
    
    features = ['pitch','intensity','timbre','location','rhythm']
    groups2 = ['controls','amusics','all']
    
    fam_stats = {}
    fam_diff = {}
    for f in features:
        fam_stats[f] = {}
        fam_diff[f] = {}
        for g in groups2:
            diff = []
            diff = all_stats[g]['familiar'][f + '_MMN'][:,time_stats_fam,:] 
            diff = diff - all_stats[g]['unfamiliar'][f + '_MMN'][:,time_stats_fam,:]
            fam_diff[f][g] = diff
            fam_stats[f][g] = mne.stats.spatio_temporal_cluster_1samp_test(fam_diff[f][g],
                                                            connectivity = con[0],
                                                            n_permutations = nperm,
                                                            out_type='mask')
            
        fam_stats[f]['interaction'] = mne.stats.spatio_temporal_cluster_test([fam_diff[f]['amusics'],
                                                      fam_diff[f]['controls']],
                                                      connectivity = con[0],
                                                      stat_fun = stat_fun_t,
                                                      threshold = 2,
                                                      n_permutations = nperm,
                                                      out_type = 'mask')
        
        fmain_f = all_stats['all']['familiar'][f + '_MMN'][:,time_stats_fam,:]
        fmain_u = all_stats['all']['unfamiliar'][f + '_MMN'][:,time_stats_fam,:]
    
        fam_stats[f]['main'] = mne.stats.spatio_temporal_cluster_test([fmain_f,
                                                      fmain_u],
                                                      connectivity = con[0],                                                  
                                                      stat_fun = stat_fun_t,
                                                      threshold = 2,
                                                      n_permutations = nperm,
                                                      out_type = 'mask')
        
    with open(out_dir + 'familiarity_stats.p', 'wb') as fp:
        pickle.dump(fam_stats, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(out_dir + 'MMN_check.p', 'rb') as fp:
          MMN_check = pickle.load(fp)    
with open(out_dir + 'complexity_stats.p', 'rb') as fp:
          out_stats = pickle.load(fp)    
with open(out_dir + 'familiarity_stats.p', 'rb') as fp:
          fam_stats = pickle.load(fp)
          
###############################################################################

############################ Make plots #######################################
chan_sel = ['Fz','F1','F2','F3','F4','FCz','FC1','FC2','FC3','FC4']

## standard, deviant, MMN (all conds in a single pdf):
conditions = ['optimal','alberti','melody','familiar','unfamiliar']
cond_names = ['LC','IC','HC','familiar','unfamiliar']  
features = ['pitch','intensity','timbre','location','rhythm']
groups = ['controls','amusics']
channs = chan_sel 
ch_names = grand_avg['controls']['optimal']['pitch_MMN'].ch_names
with PdfPages(out_dir + 'all_conds_features_and_groups_grand_average.pdf') as pdf:
    for feat_idx, feature in enumerate(features):
        
        ## Prepare figure and grid:
        f = plt.figure(figsize = (12,20))
        plt.suptitle(feature, fontsize = 20)
        gs = gridspec.GridSpec(5,4,left=0.05, right=0.98,
                               top=0.93,bottom = 0.05,
                               wspace=0.3, hspace = 0.3,
                               width_ratios =[1,3,1,3])
        for idx,group in enumerate(groups): 
            for cond_idx,cond in enumerate(conditions):
                if ((idx == 0) & (cond_idx == 0)):
                    legend = True
                else: legend = False
                topo_ax = plt.subplot(gs[cond_idx,idx*2 + 0])
                evkd_ax = plt.subplot(gs[cond_idx,idx*2 + 1]) 
                
                mp.plot_std_dev(feature = feature, group = group, cond = cond,
                               ch_names = ch_names, channs = channs, legend = legend,
                               stat_results = MMN_check, grand_avg = grand_avg,
                               all_stats = all_stats, topo_ax = topo_ax,
                               evkd_ax = evkd_ax,disp_xlab = 1,
                               disp_ylab = 1, legend_location = 4,
                               cond_name = cond_names[cond_idx],
                               colorbar = False) 
                
                if group == 'controls': hloc = 0.25 
                else: hloc = 0.75
                   
                if cond_idx == 0:
                    evkd_ax.annotate(group,xy = (hloc,0.95),xytext = (hloc,0.95),
                    xycoords = ('figure fraction','figure fraction'),
                    textcoords='offset points',
                    size=18, ha='center', va='top')                
        pdf.savefig()

plt.close('all')

##############################################################################

#### standard, deviant, MMN (complexity only - single pic layout2): ##########
        
features = ['pitch','intensity','timbre','location']#,'rhythm']
groups = ['controls','amusics']
channs = chan_sel 
ch_names = grand_avg['controls']['optimal']['pitch_MMN'].ch_names
conditions = ['optimal','alberti','melody']
cond_names = ['LC','IC','HC']

## Prepare figure and grid:
#f = plt.figure(figsize = (10,13))
f = plt.figure(figsize = (8.27,10.69))
#plt.suptitle('Complexity', fontsize = 20)
gs = gridspec.GridSpec(12,11,left=0.05, right=0.98,
                       top=0.93,bottom = 0.01,
                       wspace=0.3, hspace = 0.3,
                       width_ratios = [0.5,0.1,3,1.2,0.05,3,1.2,0.05,3,1.2,0.5],
                       height_ratios = [1,1,0.5,1,1,0.5,1,1,0.5,1,1,0.5])

feat_locs = [[0,2],[3,2],[6,2],[9,2]]
spac = 0.235

for feat_idx, feature in enumerate(features):
    floc = feat_locs[feat_idx]
    for group_idx, group in enumerate(groups):
        group_ax = plt.subplot(gs[feat_locs[feat_idx][0]+group_idx,0])
        group_ax.axis('off')
        group_ax.annotate(group, xy = (0.5,0.5),#xytext = (0.1,0.95),
                         # xycoords = ('figure fraction','figure fraction'), 
                          textcoords='offset points', size=12, ha='center',
                          va='center',rotation = 90)
        for cond_idx, cond in enumerate(conditions):

            if ((group_idx == 0) & (cond_idx == 0) & (feat_idx == 0)):
                legend = True
                colorbar = True
            else:
                legend = False
                colorbar = False 
            topo_ax = plt.subplot(gs[floc[0] + group_idx,
                                     floc[1] + cond_idx*3 + 1])
            evkd_ax = plt.subplot(gs[floc[0] + group_idx,
                                     floc[1] + cond_idx*3 + 0])
            if group_idx == 0:
               evkd_ax.axes.xaxis.set_ticklabels([])

            disp_xlab = 0
            disp_ylab = 0
            colorbar = False
            if (cond_idx == 0):
                disp_ylab = 1
            if (group_idx == 1):
                disp_xlab = 1

            mp.plot_std_dev(feature = feature, group = group, cond = cond,
                           ch_names = ch_names, channs = channs, legend = False,
                           stat_results = MMN_check, grand_avg = grand_avg,
                           all_stats = all_stats, topo_ax = topo_ax,
                           evkd_ax = evkd_ax, disp_xlab = disp_xlab,
                           disp_ylab = disp_ylab, legend_location = 4,
                           cond_name = cond_names[cond_idx],colorbar = colorbar) 
            if group_idx == 1:   
               topo_ax.set_title('')
            evkd_ax.yaxis.labelpad = -8
    evkd_ax.annotate(feature, xy = (0.12,0.96 - spac*feat_idx),
                     xytext = (0.12,0.96- spac*feat_idx),
    xycoords = ('figure fraction','figure fraction'), textcoords='offset points',
    size=16, ha='left', va='top')

## Add colorbar
cmap = mpl.cm.RdBu_r
norm = mpl.colors.Normalize(vmin=-3.1, vmax=3.1)
cax = plt.subplot(gs[0,10])
pos = cax.get_position()
new_pos = [pos.x0 - 0.015, pos.y0 + 0.02, pos.width*0.5, pos.height*0.5] 
cax.set_position(new_pos)
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical',
                                ticklocation = 'right')
cb1.ax.tick_params(labelsize=7,size = 0)
cb1.set_ticks([-3,0,3])

## Add legend
legend_elements = [mpl.lines.Line2D([0], [0], color='b', lw=4, label='Standard',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='r', lw=4, label='Deviant',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='k', lw=4, label='MMN',
                                    ls = '-')
                   ]

f.legend(handles=legend_elements, loc=[0.3,0.005],ncol = 3, edgecolor = 'black',
           shadow = True,framealpha = 1)             
plt.tight_layout()             
plt.savefig(out_dir + 'std_dev_complexity.pdf') 
plt.savefig(out_dir + 'std_dev_complexity.png', dpi = 300)

plt.close('all')

##############################################################################

#### standard, deviant, MMN (familiarity only - single pic layout2): #########
        
features = ['pitch','intensity','timbre','location']#,'rhythm']
groups = ['controls','amusics']
channs = chan_sel 
ch_names = grand_avg['controls']['optimal']['pitch_MMN'].ch_names
conditions = ['familiar','unfamiliar']
cond_names = ['familiar','unfamiliar']

## Prepare figure and grid:
#f = plt.figure(figsize = (10,13))
f = plt.figure(figsize = (8.27,11))
#plt.suptitle('Complexity', fontsize = 20)
gs = gridspec.GridSpec(12,8,left=0.05, right=0.98,
                       top=0.93,bottom = 0.01,
                       wspace=0.3, hspace = 0.3,
                       width_ratios = [0.5,0.1,3,1.2,0.05,3,1.2,0.5],
                       height_ratios = [1,1,0.5,1,1,0.5,1,1,0.5,1,1,0.5])

feat_locs = [[0,2],[3,2],[6,2],[9,2]]
spac = 0.235

for feat_idx, feature in enumerate(features):
    floc = feat_locs[feat_idx]
    for group_idx, group in enumerate(groups):
        group_ax = plt.subplot(gs[feat_locs[feat_idx][0]+group_idx,0])
        group_ax.axis('off')
        group_ax.annotate(group, xy = (0.5,0.5),#xytext = (0.1,0.95),
                         # xycoords = ('figure fraction','figure fraction'), 
                          textcoords='offset points', size=12, ha='center',
                          va='center',rotation = 90)
        for cond_idx, cond in enumerate(conditions):

            if ((group_idx == 0) & (cond_idx == 0) & (feat_idx == 0)):
                legend = True
                colorbar = True
            else:
                legend = False
                colorbar = False 
            topo_ax = plt.subplot(gs[floc[0] + group_idx,
                                     floc[1] + cond_idx*3 + 1])
            evkd_ax = plt.subplot(gs[floc[0] + group_idx,
                                     floc[1] + cond_idx*3 + 0])
            if group_idx == 0:
               evkd_ax.axes.xaxis.set_ticklabels([])

            disp_xlab = 0
            disp_ylab = 0
            colorbar = False
            if (cond_idx == 0):
                disp_ylab = 1
            if (group_idx == 1):
                disp_xlab = 1

            mp.plot_std_dev(feature = feature, group = group, cond = cond,
                           ch_names = ch_names, channs = channs, legend = False,
                           stat_results = MMN_check, grand_avg = grand_avg,
                           all_stats = all_stats, topo_ax = topo_ax,
                           evkd_ax = evkd_ax, disp_xlab = disp_xlab,
                           disp_ylab = disp_ylab, legend_location = 4,
                           cond_name = cond_names[cond_idx],colorbar = colorbar) 
            if group_idx == 1:   
               topo_ax.set_title('')

            evkd_ax.yaxis.labelpad = -8
    evkd_ax.annotate(feature, xy = (0.12,0.96 - spac*feat_idx),
                     xytext = (0.12,0.96- spac*feat_idx),
    xycoords = ('figure fraction','figure fraction'), textcoords='offset points',
    size=16, ha='left', va='top')

## Add colorbar
cmap = mpl.cm.RdBu_r
norm = mpl.colors.Normalize(vmin=-3.1, vmax=3.1)
cax = plt.subplot(gs[0,7])
pos = cax.get_position()
new_pos = [pos.x0 - 0.015, pos.y0 + 0.02, pos.width*0.5, pos.height*0.5] 
cax.set_position(new_pos)
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical',
                                ticklocation = 'right')
cb1.ax.tick_params(labelsize=7,size = 0)
cb1.set_ticks([-3,0,3])

## Add legend
legend_elements = [mpl.lines.Line2D([0], [0], color='b', lw=4, label='Standard',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='r', lw=4, label='Deviant',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='k', lw=4, label='MMN',
                                    ls = '-')
                   ]

f.legend(handles=legend_elements, loc=[0.3,0.005],ncol = 3, edgecolor = 'black',
           shadow = True,framealpha = 1)             
plt.tight_layout()             
plt.savefig(out_dir + 'std_dev_familiarity.pdf') 
plt.savefig(out_dir + 'std_dev_familiarity.png', dpi = 300) 

plt.close('all')

##############################################################################

############### Main effects / complexity and familiarity ####################

features = ['pitch','intensity','timbre','location']
groups = ['controls','amusics']
channs = chan_sel 
ch_names = grand_avg['controls']['optimal']['pitch_MMN'].ch_names
effects = {'complexity': out_stats, 'familiarity': fam_stats}
effect_names = ['complexity','familiarity']
e_conds = [['optimal','alberti','melody'],['familiar','unfamiliar']]
cond_names = [['Low','Intermediate','High'],['familiar','unfamiliar']]
colors = [['k-','b-','r-'],['b-','r-']]

ntests = 4
f = plt.figure(figsize = (8,10))
plt.suptitle('Appendix 4 - Main effects', fontsize = 16)
gs = gridspec.GridSpec(4,2,left=0.1, right=0.9,
                       top=0.93,bottom = 0.05,
                       wspace=0.3, hspace = 0.3,
                       width_ratios =[1,1])

for feature_idx,feature in enumerate(features):
    for effect_idx, effect in enumerate(effect_names):
        
        ### get stats and plotting masks
        ch_indx = [iidx for iidx,cch in enumerate(ch_names) 
                   if (cch in chan_sel)]
        cur_stats = effects[effect][feature]['main']
        t_vals = cur_stats[0]
        sig_clusters_idx = np.where(cur_stats[2] < .1/ntests)[0]
        masks = [cur_stats[1][x] for x in sig_clusters_idx]
        pvals = cur_stats[2][sig_clusters_idx]
        pval_char = []
        clust_dir = np.zeros(pvals.shape)
        cluster_t = np.zeros(len(masks))
        if len(masks) > 0:
            for idxx,m in enumerate(masks):
                p = pvals[idxx]*ntests
                cluster_t[idxx] = np.sum(cur_stats[0][m])
                if p < .001:
                    p_char = 'p < 0.001'
                else:
                    p_char = 'p = ' + str(np.round(p,decimals = 3))
                pval_char.append([p_char])    
                if np.sum(t_vals*m) < 0:
                    clust_dir[idxx] = -1               
                elif np.sum(t_vals*m) > 0:
                    clust_dir[idxx] = 1
                if p > .05:
                    masks[idxx][:,:] = 0
        elif len(cur_stats[2]) > 0:
             p_char = 'p = ' + str(np.round(np.min(cur_stats[2]),decimals = 3))
             pval_char.append([p_char])
        else:
             p_char = 'N.C.F.'
             pval_char.append([p_char])
             
        ### prepare data
        conds = e_conds[effect_idx]
        erps = {}
        for cc in (conds):
            erps[cc] = grand_avg['all'][cc][feature + '_MMN'].copy()
            erps[cc] = np.mean(erps[cc].pick_channels(channs).data,0)*1e6

        time = grand_avg['all']['optimal'][feature + '_MMN'].times*1000
        
        ## prepare masks with time corrections
        mask_time_idx = np.where((time >= 0) & (time <= 300))[0]

        mask_evkd = []
        mask_topo = []
        
        for idxx, m in enumerate(masks):
            mask_evkd.append(np.zeros(erps[cc].shape[0], dtype=bool))        
            mask_evkd[idxx][mask_time_idx] = np.sum(m[:,ch_indx],1) > 0               
                   
        ### prepare axes
        evkd_ax = plt.subplot(gs[feature_idx,effect_idx])
        #evkd_ax = plt.subplot(gs[feature_idx,1])
        
        ## plot evoked
        evkd_ax.set_xlim(left = -100, right=400)   
        evkd_ax.set_ylim(bottom=-6, top=6)        
        evkd_ax.hlines(0,xmin = -100,xmax = 400) 
        evkd_ax.vlines(0,ymin = -6,ymax = 6,linestyles = 'solid')        
        
        for c_idx, cc in enumerate(conds):
            evkd_ax.plot(time,erps[cc],colors[effect_idx][c_idx],
                         label = cond_names[effect_idx][c_idx])

        
        evkd_ax.set_xlabel('ms',labelpad = 0) 
        evkd_ax.set_ylabel(r'$\mu$V',labelpad = 0)
        evkd_ax.annotate(feature,xy = (0,0), xytext = (120,5))
        
        for idxx, em in enumerate(mask_evkd):
            if clust_dir[idxx] == -1:
               color = 'b'
            else:
               color = 'r'
            evkd_ax.fill_between(time[em],-6,6, color = color,alpha = .1)  
            
        for idxx, pv in enumerate(pval_char):
            evkd_ax.annotate(pv[0],xy = (0,0), xytext = (270,4-idxx*1))
            
        if effect == 'complexity': hloc = 0.25 
        else: hloc = 0.75
           
        if feature_idx == 0:
           evkd_ax.annotate(effect,xy = (hloc,0.95),xytext = (hloc,0.95),
           xycoords = ('figure fraction','figure fraction'),
           textcoords='offset points',
           size=14, ha='center', va='top')
             
        if (feature_idx == 0):
           evkd_ax.legend(fontsize = 8, loc = 2, framealpha = 1,
                          edgecolor = 'black',shadow = True)
         
        plt.tight_layout()
plt.savefig(out_dir + 'appendix 4 - main_effects.pdf')
plt.close('all')

##############################################################################

##################### simple effects: Complexity #############################
features = ['pitch','intensity','timbre','location']
groups = ['controls','amusics']
channs = chan_sel 
ch_names = grand_avg['controls']['optimal']['pitch_MMN'].ch_names
ntests = 4

f = plt.figure(figsize = (8.27,9.69))
#plt.suptitle('Complexity', fontsize = 18)
gs = gridspec.GridSpec(5,3,left=0.05, right=0.9,
                       top=0.93,bottom = 0.05,
                       wspace=0.3, hspace = 0.3,
                       width_ratios =[0.05,1,1],
                       height_ratios = [1,1,1,1,0.0005])
 
for feature_idx,feature in enumerate(features):
    for group_idx, group in enumerate(groups):
        evkd_ax = plt.subplot(gs[feature_idx,group_idx+1])
        feature_ax = plt.subplot(gs[feature_idx,0])
        feature_ax.axis('off')
        feature_ax.annotate(feature, xy = (0.5,0.5),#xytext = (0.1,0.95),
                         # xycoords = ('figure fraction','figure fraction'), 
                          textcoords='offset points', size=12, ha='center',
                          va='center',rotation = 90)
        if ((feature_idx == 0) & (group_idx == 0)):
            legend = True
        else: legend = False
         
        mp.simpleComplexity(feature = feature, group = group, 
                            ch_names = ch_names, channs = channs,
                            stat_results = out_stats, ntests = ntests,
                            grand_avg = grand_avg, evkd_ax = evkd_ax,
                            legend = False)
        
        if group == 'controls': hloc = 0.2 + 0.1
        else: hloc = 0.62 + 0.1
           
        if feature_idx == 0:
           evkd_ax.annotate(group,xy = (hloc,0.97),xytext = (hloc,0.97),
           xycoords = ('figure fraction','figure fraction'),
           textcoords='offset points',
           size=16, ha='center', va='top')
           
        if group_idx == 1:   
           evkd_ax.set_ylabel('')
        evkd_ax.yaxis.labelpad = -8
        
legend_elements = [mpl.lines.Line2D([0], [0], color='k', lw=4, label='low',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='b', lw=4, label='Intermediate',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='r', lw=4, label='High',
                                    ls = '-')
                   ]
f.legend(handles=legend_elements, loc=[0.3,0.005],ncol = 3, edgecolor = 'black',
           shadow = True,framealpha = 1) 
plt.tight_layout()
plt.savefig(out_dir + 'complexity.pdf')
plt.savefig(out_dir + 'complexity.png', dpi = 300)

plt.close('all')

####################### Complexity interactions #############################
features = ['pitch','intensity','timbre','location']
groups = ['controls','amusics']
channs = chan_sel 
ch_names = grand_avg['controls']['optimal']['pitch_MMN'].ch_names
pairs = ['ao','mo','ma']
pairs2 = [['optimal','alberti'],['optimal','melody'],['alberti','melody']]
pairs3 = [['low','int'],['low','high'],['int','high']]
ntests = 4
f = plt.figure(figsize = (14,12))
plt.suptitle('Complexity x group interaction (difference of differences)', fontsize = 16)
gs = gridspec.GridSpec(4,6,left=0.03, right=0.98,
                       top=0.93,bottom = 0.05,
                       wspace=0.3, hspace = 0.3,
                       width_ratios =[1,3,1,3,1,3])

for pidx, pair in enumerate(pairs):
    pair2 = pairs2[pidx]
    for feature_idx,feature in enumerate(features):
            if ((feature_idx == 0) & (pidx == 0)):
                legend = True
            else: legend = False
            
            topo_ax = plt.subplot(gs[feature_idx,0 + 2*pidx])
            evkd_ax = plt.subplot(gs[feature_idx,1 + 2*pidx])
                
            mp.interComplexity(feature = feature, groups = groups, pair = pair,
                               pair2 = pair2, ch_names = ch_names, channs = channs,
                               stat_results = out_stats, ntests = ntests,
                               grand_avg = grand_avg, all_stats = all_stats,
                               topo_ax = topo_ax, evkd_ax = evkd_ax, legend = legend)    
            
            if pidx == 0: hloc = 0.25 
            elif pidx == 1: hloc = 0.57
            else: hloc =  0.89
            
            if feature_idx == 0:
               evkd_ax.annotate(pairs3[pidx][0] + ' - ' + pairs3[pidx][1],
                                xy = (hloc,0.95),xytext = (hloc,0.95),
               xycoords = ('figure fraction','figure fraction'),
               textcoords='offset points',
               size=16, ha='center', va='top')           
                     
            plt.tight_layout()
plt.savefig(out_dir + 'complexity_interaction.pdf')
plt.close('all')

###############################################################################

################### simple effects: familiarity ###############################
  
ntests = 1 # we will show results before correcton, and clarify in caption
features = ['pitch','intensity','timbre','location']
groups = ['controls','amusics']
channs = chan_sel 
ch_names = grand_avg['controls']['optimal']['pitch_MMN'].ch_names

f = plt.figure(figsize = (8.27,9.69))
#plt.suptitle('Familiarity', fontsize = 18)
gs = gridspec.GridSpec(5,5,left=0.05, right=0.98,
                       top=0.93,bottom = 0.001,
                       wspace=0.3, hspace = 0.3,
                       width_ratios =[3,1,3,1,0.3],
                       height_ratios = [1,1,1,1,0.2])

for feature_idx,feature in enumerate(features):
    for group_idx, group in enumerate(groups):
        if ((feature_idx == 0) & (group_idx == 0)):
            legend = True
        else: legend = False
        
        topo_ax = plt.subplot(gs[feature_idx,group_idx*2 + 1])
        evkd_ax = plt.subplot(gs[feature_idx,group_idx*2 + 0])
        
        mp.simpleFamiliarity(feature = feature, group = group, ch_names = ch_names,
                             channs = channs, stat_results = fam_stats, ntests = ntests,
                             grand_avg = grand_avg, all_stats = all_stats,
                             evkd_ax = evkd_ax, topo_ax = topo_ax, legend = False)
        
        if group == 'controls': hloc = 0.1
        else: hloc = 0.53
           
        if feature_idx == 0:
           evkd_ax.annotate(groups[group_idx],xy = (hloc,0.96),xytext = (hloc,0.96),
           xycoords = ('figure fraction','figure fraction'),
           textcoords='offset points',
           size=14, ha='center', va='top')
        if group_idx == 1:   
           evkd_ax.set_ylabel('')
        if feature_idx != 3:   
           evkd_ax.set_xlabel('') 
        evkd_ax.yaxis.labelpad = -8
        
cmap = mpl.cm.RdBu_r
norm = mpl.colors.Normalize(vmin=-3.1, vmax=3.1)
cax = plt.subplot(gs[0,4])
pos = cax.get_position()
new_pos = [pos.x0 - 0.015, pos.y0 + 0.03, pos.width*0.5, pos.height*0.5] 
cax.set_position(new_pos)
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical',
                                ticklocation = 'right')
cb1.ax.tick_params(labelsize=10,size = 0)
cb1.set_ticks([-3,0,3])

legend_elements = [mpl.lines.Line2D([0], [0], color='b', lw=4, label='Familiar',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='r', lw=4, label='Unfamiliar',
                                    ls = '--'),
                   mpl.lines.Line2D([0], [0], color='k', lw=4, label='Difference',
                                    ls = '-')
                   ]

f.legend(handles=legend_elements, loc=[0.2,0.005],ncol = 3, edgecolor = 'black',
           shadow = True,framealpha = 1) 

plt.tight_layout()        
plt.savefig(out_dir + 'familiarity.pdf')
plt.savefig(out_dir + 'familiarity.png', dpi = 300)
plt.close('all')        
                
########################## Familiarity interaction ##########################
ntests = 4
features = ['pitch','intensity','timbre','location']
groups = ['controls','amusics']
channs = chan_sel 
ch_names = grand_avg['controls']['optimal']['pitch_MMN'].ch_names

f = plt.figure(figsize = (14,10))
plt.suptitle('Interaction (Group x Familiarity)',fontsize = 16)
gs = gridspec.GridSpec(2,4,left=0.05, right=0.95,
                       top=0.93,bottom = 0.05,
                       wspace=0.3, hspace = 0.5,
                       width_ratios =[1,3,1,3])

topo_axes = [[0,0], [0,2], [1,0], [1,2]]
evkd_axes = [[0,1], [0,3], [1,1], [1,3]]
for feature_idx,feature in enumerate(features):
        topo_ax = plt.subplot(gs[topo_axes[feature_idx][0],topo_axes[feature_idx][1]])
        evkd_ax = plt.subplot(gs[evkd_axes[feature_idx][0],evkd_axes[feature_idx][1]])
        if feature_idx == 0:
            legend = True
        else: legend = False
        
        mp.interFamiliarity(feature = feature, groups = groups, ch_names = ch_names,
                            channs = channs, stat_results = fam_stats, 
                            grand_avg = grand_avg, all_stats = all_stats, 
                            topo_ax = topo_ax, evkd_ax = evkd_ax, legend = legend,
                            ntests = ntests)
               
plt.tight_layout()
plt.savefig(out_dir + 'familiarity_interaction.pdf')
plt.close('all')

########################### Rhythm plots #####################################
# groups = ['controls','amusics']
# ntests = 1
# channs = chan_sel 
# ch_names = grand_avg['controls']['optimal']['pitch_MMN'].ch_names

# pairs = ['ao','mo','ma']
# pairs2 = [['optimal','alberti'],['optimal','melody'],['alberti','melody']]
# pairs3 = [['low','int'],['low','high'],['int','high']]

# f = plt.figure(figsize = (10,10))
# plt.suptitle('Effects on the rhythm MMN',fontsize = 14)
# gs = gridspec.GridSpec(4,4,left=0.03, right=0.97,
#                        top=0.93,bottom = 0.05,
#                        wspace=0.3, hspace = 0.5,
#                        width_ratios =[1,3,1,3])        

# topo_ax = plt.subplot(gs[0,0])
# evkd_ax = plt.subplot(gs[0,1])
# size = 14
# hloc = 0.33
# vloc = 0.95
# hfactor = 0.5
# vfactor = -0.24

# mp.simpleFamiliarity(feature = 'rhythm', group = 'controls', ch_names = ch_names,
#                      channs = channs, stat_results = fam_stats, ntests = ntests,
#                      grand_avg = grand_avg, all_stats = all_stats,
#                      evkd_ax = evkd_ax, topo_ax = topo_ax, legend = True)

# evkd_ax.annotate('familiarity - controls',xy = (hloc,vloc),xytext = (hloc,vloc),
#                  xycoords = ('figure fraction','figure fraction'),
#                  textcoords='offset points',
#                  size=size, ha='center', va='top')

# topo_ax = plt.subplot(gs[0,2])
# evkd_ax = plt.subplot(gs[0,3])
       
# mp.simpleFamiliarity(feature = 'rhythm', group = 'amusics', ch_names = ch_names,
#                      channs = channs, stat_results = fam_stats, ntests = ntests,
#                      grand_avg = grand_avg, all_stats = all_stats,
#                      evkd_ax = evkd_ax, topo_ax = topo_ax, legend = True)

# evkd_ax.annotate('familiarity - amusics', xy = (hloc + hfactor,vloc),
#                  xytext = (hloc + hfactor ,0.95),
#                  xycoords = ('figure fraction','figure fraction'),
#                  textcoords='offset points',
#                  size=size, ha='center', va='top')

# topo_ax = plt.subplot(gs[1,0])
# evkd_ax = plt.subplot(gs[1,1])
# mp.interFamiliarity(feature = 'rhythm', groups = groups, ch_names = ch_names,
#                     channs = channs, stat_results = fam_stats, 
#                     grand_avg = grand_avg, all_stats = all_stats, 
#                     topo_ax = topo_ax, evkd_ax = evkd_ax, legend = True,
#                     ntests = ntests)

# evkd_ax.annotate('familiarity - interaction',xy = (hloc,vloc + vfactor),
#                  xytext = (hloc,vloc + vfactor),
#                  xycoords = ('figure fraction','figure fraction'),
#                  textcoords='offset points',
#                  size=size, ha='center', va='top')

# evkd_ax = plt.subplot(gs[1,3])

# mp.simpleComplexity(feature = 'rhythm', group = 'controls', 
#                     ch_names = ch_names, channs = channs,
#                     stat_results = out_stats, ntests = ntests,
#                     grand_avg = grand_avg, evkd_ax = evkd_ax,
#                     legend = True)

# evkd_ax.annotate('complexity - controls',xy = (hloc + hfactor, vloc + vfactor),
#                  xytext = (hloc + hfactor,vloc + vfactor),
#                  xycoords = ('figure fraction','figure fraction'),
#                  textcoords='offset points',
#                  size=size, ha='center', va='top')

# evkd_ax = plt.subplot(gs[2,1])

# mp.simpleComplexity(feature = 'rhythm', group = 'controls', 
#                     ch_names = ch_names, channs = channs,
#                     stat_results = out_stats, ntests = ntests,
#                     grand_avg = grand_avg, evkd_ax = evkd_ax,
#                     legend = True)


# evkd_ax.annotate('complexity - amusics',xy = (hloc,vloc + vfactor*2),
#                  xytext = (hloc,vloc + vfactor),
#                  xycoords = ('figure fraction','figure fraction'),
#                  textcoords='offset points',
#                  size=size, ha='center', va='top')

# topo_ax = plt.subplot(gs[2,2])
# evkd_ax = plt.subplot(gs[2,3])

# pair = 'ao'
# pair2 = ['optimal','alberti']
# mp.interComplexity(feature = 'rhythm', groups = groups, pair = pair,
#                    pair2 = pair2, ch_names = ch_names, channs = channs,
#                    stat_results = out_stats, ntests = ntests,
#                    grand_avg = grand_avg, all_stats = all_stats,
#                    topo_ax = topo_ax, evkd_ax = evkd_ax, legend = True)  

# evkd_ax.annotate('complexity - int vs low',xy = (hloc + hfactor, vloc + vfactor*2),
#                  xytext = (hloc + hfactor, vloc + vfactor),
#                  xycoords = ('figure fraction','figure fraction'),
#                  textcoords='offset points',
#                  size=size, ha='center', va='top')


# topo_ax = plt.subplot(gs[3,0])
# evkd_ax = plt.subplot(gs[3,1])
# pair = 'mo'
# pair2 = ['optimal','melody']
# mp.interComplexity(feature = 'rhythm', groups = groups, pair = pair,
#                    pair2 = pair2, ch_names = ch_names, channs = channs,
#                    stat_results = out_stats, ntests = ntests,
#                    grand_avg = grand_avg, all_stats = all_stats,
#                    topo_ax = topo_ax, evkd_ax = evkd_ax, legend = True)

# evkd_ax.annotate('complexity - high vs low',xy = (hloc,vloc + vfactor*3),
#                  xytext = (hloc,vloc + vfactor),
#                  xycoords = ('figure fraction','figure fraction'),
#                  textcoords='offset points',
#                  size=size, ha='center', va='top')


# topo_ax = plt.subplot(gs[3,2])
# evkd_ax = plt.subplot(gs[3,3])
# pair = 'ma'
# pair2 = ['alberti','melody']
# mp.interComplexity(feature = 'rhythm', groups = groups, pair = pair,
#                    pair2 = pair2, ch_names = ch_names, channs = channs,
#                    stat_results = out_stats, ntests = ntests,
#                    grand_avg = grand_avg, all_stats = all_stats,
#                    topo_ax = topo_ax, evkd_ax = evkd_ax, legend = True)

# evkd_ax.annotate('complexity - high vs int',xy = (hloc + hfactor, vloc + vfactor*3),
#                  xytext = (hloc + hfactor, vloc + vfactor),
#                  xycoords = ('figure fraction','figure fraction'),
#                  textcoords='offset points',
#                  size=size, ha='center', va='top')

# plt.tight_layout()
# plt.savefig(out_dir + 'rhythm_effects.pdf')
# plt.close('all')

          












