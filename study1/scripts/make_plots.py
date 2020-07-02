# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:50:38 2020

@author: au571303
"""
import numpy as np
#import matplotlib.pyplot as plt

def plot_std_dev(feature,group,cond,ch_names,channs,stat_results,
                 grand_avg,all_stats,topo_ax,evkd_ax,legend,
                 disp_xlab, disp_ylab, cond_name, legend_location,
                 colorbar):
    
    ### Get statistics:
    ch_indx = [iidx for iidx,cch in enumerate(ch_names)
                           if (cch in channs)]
    cur_stats = stat_results[feature + '_MMN'][group][cond]
    t_vals = cur_stats[0]
    sig_clusters_idx = np.where(cur_stats[2] < .05)[0]

    masks = [cur_stats[1][x] for x in sig_clusters_idx]
    pvals = cur_stats[2][sig_clusters_idx]
    neg_masks = []
    neg_tvals = []
    neg_pvals = []
    for idxx,m in enumerate(masks):
        if np.sum(t_vals*m) < 0:
           neg_masks.append(m)
           neg_tvals.append([np.sum(t_vals*m)])
           neg_pvals.append(pvals[idxx])
    if neg_tvals:
        midx = np.argmin(neg_tvals)
        mask_evkd_prov = np.sum(neg_masks[midx][:,ch_indx],1)>0
        mask_topo_prov = np.transpose(neg_masks[midx],(1,0))
        MMN_pval = neg_pvals[midx]
        if MMN_pval < .001:
            MMN_pval = 'p < 0.001'
        else:
            MMN_pval = 'p = ' + str(np.round(MMN_pval,decimals = 3))
    elif len(cur_stats[2]) > 0:
        MMN_pval = 'p = ' + str(np.round(np.min(cur_stats[2]),decimals = 3))
    else:
        midx = []                
        MMN_pval = 'N.C.F.'
    
    #prepare data 
    if feature == 'rhythm':       
        std = grand_avg[group][cond]['standard_rhy'].copy()
    else:
        std = grand_avg[group][cond]['standard'].copy()
        
    std = np.mean(std.pick_channels(channs).data,0)*1e6
    dev = grand_avg[group][cond][feature].copy()
    dev = np.mean(dev.pick_channels(channs).data,0)*1e6
    MMN = grand_avg[group][cond][feature + '_MMN'].copy()
    MMN = np.mean(MMN.pick_channels(channs).data,0)*1e6
    time = grand_avg[group][cond][feature].times*1000
    
    ## prepare masks with time corrections
    mask_time_idx = np.where((time >= 0) & (time <= 300))[0]
    mask_evkd = np.zeros(MMN.data.shape[0], dtype=bool)
    mask_topo = np.zeros(grand_avg[group][cond][feature + '_MMN'].data.shape,
                         dtype=bool)
    if neg_tvals:
       mask_evkd[mask_time_idx] = mask_evkd_prov               
       mask_topo[:,mask_time_idx] = mask_topo_prov
       
    ### Get 95% CI
    se_data = all_stats[group][cond][feature + '_MMN'][:,:,ch_indx]
    stdev = np.std(np.mean(se_data,2),0)
    se = stdev/np.sqrt(se_data.shape[0])
    ci_upper = MMN + 1.96*1e6*se
    ci_lower = MMN - 1.96*1e6*se
    
    ## plot evoked
    evkd_ax.set_xlim(left = -100, right=400)   
    evkd_ax.set_ylim(bottom=-7, top=7)        
    evkd_ax.hlines(0,xmin = -100,xmax = 400) 
    evkd_ax.vlines(0,ymin = -7,ymax = 7,linestyles = 'solid')        
    
    for isp in range(0,se_data.shape[0]):
        isp_data =  np.mean(se_data[isp,:,:],1)*1e6
        evkd_ax.plot(time,isp_data,'k-',alpha = .15,
                     linewidth = 0.5)
        
    evkd_ax.plot(time,std,'b--',label = 'standard')
    evkd_ax.plot(time,dev,'r--',label = 'deviant')
    evkd_ax.plot(time,MMN,'k-',label = 'MMN')
    if disp_xlab:
       evkd_ax.set_xlabel('ms',labelpad = 0) 
    if disp_ylab:
       evkd_ax.set_ylabel(r'$\mu$V',labelpad = 0) 
    evkd_ax.fill_between(time[mask_evkd],-7,7, color = 'b',alpha = .1)
    evkd_ax.fill_between(time,ci_lower,ci_upper, color = 'k',alpha = .2)   
    evkd_ax.annotate(MMN_pval,xy = (0,0), xytext = (10,5),size=8)
                  
    if legend:
       evkd_ax.legend(fontsize = 6, loc = legend_location, framealpha = 1,
                      edgecolor = 'black',shadow = True)
    cur_data = MMN.copy()
    peak_idx = ((cur_data[1:-1] - cur_data[2:] < 0) &
               (cur_data[1:-1] - cur_data[0:-2] < 0) &
               (time[1:-1] >= 0) & (time[1:-1] <= 300))
    cur_data = cur_data[1:-1]
    peak_idx2 = np.where((peak_idx == True) &
                        (cur_data == np.amin(cur_data[peak_idx])))
    lat = time[peak_idx2]/1000
    
    ## topoplot
    mask_params = dict(marker='o',
                   markerfacecolor='r',
                   markeredgecolor='k',
                   linewidth=0,
                   markersize=2)
    grand_avg[group][cond][feature + '_MMN'].plot_topomap(times = lat,
                                                          axes = topo_ax,
                                                          vmin = -3,
                                                          vmax = 3,
                                                          colorbar = colorbar,
                                                          mask = mask_topo,
                                                          mask_params = mask_params,
                                                          sensors = False)  
    
    topo_ax.set_title(cond_name,fontsize = 14)
    disp_lat = str(int(np.round(lat[0]*1000,0))) + ' ms'
    topo_ax.annotate(disp_lat,xy = (0.5,-0.5),xytext = (0.5,-0.5),
    xycoords = ('axes fraction','axes fraction'),
    textcoords='offset points',
    size=10, ha='center', va='bottom') 

##############################################################################   
def simpleComplexity(feature,group,ch_names,channs,stat_results,ntests,
                     grand_avg, evkd_ax, legend):
    
    ch_indx = [iidx for iidx,cch in enumerate(ch_names) 
               if (cch in channs)]
    cur_stats = stat_results[feature][group]['Ftest']
    t_vals = cur_stats[0]
    sig_clusters_idx = np.where(cur_stats[2] < .05/ntests)[0]
    masks = [cur_stats[1][x] for x in sig_clusters_idx]
    pvals = cur_stats[2][sig_clusters_idx]
    pval_char = []
    clust_dir = np.zeros(pvals.shape)
    cluster_t = np.zeros(len(masks))
    if len(masks) > 0:
        for idxx,m in enumerate(masks):
            p = pvals[idxx]
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
    elif len(cur_stats[2]) > 0:
         p_char = 'p = ' + str(np.round(np.min(cur_stats[2]),decimals = 3))
         pval_char.append([p_char])
    else:
         p_char = 'N.C.F.'
         pval_char.append([p_char])   
              

    opt = grand_avg[group]['optimal'][feature + '_MMN'].copy()
    opt = np.mean(opt.pick_channels(channs).data,0)*1e6
    alb = grand_avg[group]['alberti'][feature + '_MMN'].copy()
    alb = np.mean(alb.pick_channels(channs).data,0)*1e6
    mel = grand_avg[group]['melody'][feature + '_MMN'].copy()
    mel = np.mean(mel.pick_channels(channs).data,0)*1e6

    time = grand_avg[group]['optimal'][feature + '_MMN'].times*1000
    
    ## prepare masks with time corrections
    mask_time_idx = np.where((time >= 0) & (time <= 300))[0]

    mask_evkd = []
    
    for idxx, m in enumerate(masks):
        mask_evkd.append(np.zeros(mel.shape[0], dtype=bool))        
        mask_evkd[idxx][mask_time_idx] = np.sum(m[:,ch_indx],1) > 0
        
    evkd_ax.set_xlim(left = -100, right=400)   
    evkd_ax.set_ylim(bottom=-6, top=6)        
    evkd_ax.hlines(0,xmin = -100,xmax = 400) 
    evkd_ax.vlines(0,ymin = -6,ymax = 6,linestyles = 'solid')        
               
    evkd_ax.plot(time,opt,'k-',label = 'low')
    evkd_ax.plot(time,alb,'b-',label = 'int.')
    evkd_ax.plot(time,mel,'r-',label = 'high')
    evkd_ax.set_xlabel('ms',labelpad = 0) 
    evkd_ax.set_ylabel(r'$\mu$V',labelpad = 0)
#    evkd_ax.annotate(feature,xy = (0,0), xytext = (120,5))
                 
    for idxx, em in enumerate(mask_evkd):
        if clust_dir[idxx] == -1:
           color = 'b'
        else:
           color = 'r'
        evkd_ax.fill_between(time[em],-6,6, color = color,alpha = .1)
    for idxx, pv in enumerate(pval_char):
        evkd_ax.annotate(pv[0],xy = (0,0), xytext = (270,4-idxx*1))
    
    if legend:
       evkd_ax.legend(fontsize = 8, loc = 2, framealpha = 1,
                      edgecolor = 'black',shadow = True)

def interComplexity(feature,groups,pair,pair2,ch_names,channs,stat_results,ntests,
                    grand_avg,all_stats,topo_ax,evkd_ax,legend):
            pnm1 = pair2[0]
            pnm2 = pair2[1]
            ### get stats and plotting masks
            ch_indx = [iidx for iidx,cch in enumerate(ch_names) 
                       if (cch in channs)]
            cur_stats = stat_results[feature]['interaction'][pair]
            t_vals = cur_stats[0]
            sig_clusters_idx = np.where(cur_stats[2] < .05/ntests)[0]
            masks = [cur_stats[1][x] for x in sig_clusters_idx]
            pvals = cur_stats[2][sig_clusters_idx]
            pval_char = []
            clust_dir = np.zeros(pvals.shape)
            cluster_t = np.zeros(len(masks))
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
                
            ### prepare data
    
            amus = (grand_avg['amusics'][pnm1][feature + '_MMN'].data - 
                    grand_avg['amusics'][pnm2][feature + '_MMN'].data) 
            amus = np.mean(amus[ch_indx,:],0)*1e6
            cont = (grand_avg['controls'][pnm1][feature + '_MMN'].data -
                    grand_avg['controls'][pnm2][feature + '_MMN'].data)
            cont = np.mean(cont[ch_indx,:],0)*1e6
            diff = amus - cont
            time = grand_avg['amusics']['optimal'][feature + '_MMN'].times*1000
            
            ## prepare masks with time corrections
            mask_time_idx = np.where((time >= 0) & (time <= 300))[0]
    
            mask_evkd = []
            mask_topo = []
            
            for idxx, m in enumerate(masks):
                mask_evkd.append(np.zeros(diff.shape[0], dtype=bool))
                mask_topo.append(np.zeros(grand_avg['amusics']['optimal'][feature + '_MMN'].data.shape,
                                     dtype=bool))
            
                mask_evkd[idxx][mask_time_idx] = np.sum(m[:,ch_indx],1) > 0               
                mask_topo[idxx][:,mask_time_idx] = np.transpose(m,(1,0))
                
            ### Get 95% CI
            ci = {}
            for group in groups:
                se_data = (all_stats[group][pnm1][feature + '_MMN'][:,:,ch_indx] -
                           all_stats[group][pnm2][feature + '_MMN'][:,:,ch_indx])
            
                stdev = np.std(np.mean(se_data,2),0)
                se = stdev/np.sqrt(se_data.shape[0])
                ci[group] = 1.96*1e6*se   
                    
            ## plot evoked
            evkd_ax.set_xlim(left = -100, right=400)   
            evkd_ax.set_ylim(bottom=-4, top=4)        
            evkd_ax.hlines(0,xmin = -100,xmax = 400) 
            evkd_ax.vlines(0,ymin = -4,ymax = 4,linestyles = 'solid')        
            evkd_ax.plot(time,amus,'b-',label = 'amusics')
            evkd_ax.plot(time,cont,'r-',label = 'controls')
            evkd_ax.plot(time,diff,'k--',label = 'difference')
            evkd_ax.set_xlabel('ms',labelpad = 0) 
            evkd_ax.set_ylabel(r'$\mu$V',labelpad = 0)
            evkd_ax.fill_between(time,amus + ci['amusics'],
                                 amus - ci['amusics'], color = 'b',alpha = .2)
            evkd_ax.fill_between(time,cont + ci['controls'],
                                 cont - ci['controls'], color = 'r',alpha = .2)
            
            if not mask_evkd:
               evkd_ax.annotate('N.S.',xy = (0,0), xytext = (270,3))
               
            for idxx, em in enumerate(mask_evkd):
                if clust_dir[idxx] == -1:
                   color = 'r'
                else:
                   color = 'b'
                evkd_ax.fill_between(time[em],-4,4, color = color,alpha = .1)            
                evkd_ax.annotate(pval_char[idxx][0],xy = (0,0), xytext = (270,3-1*idxx))
            
               
            if legend:
               evkd_ax.legend(fontsize = 8, loc = 2, framealpha = 1,
                              edgecolor = 'black',shadow = True)
            
            ## get peak latency of difference
            cur_data = np.abs(diff.copy())
            peak_idx = ((cur_data[1:-1] - cur_data[2:] > 0) &
                      (cur_data[1:-1] - cur_data[0:-2] > 0) &
                      (time[1:-1] >= 0) & (time[1:-1] <= 300))
            cur_data = cur_data[1:-1]
            peak_idx2 = np.where((peak_idx == True) &
                                (cur_data == np.amax(cur_data[peak_idx])))
            lat = time[peak_idx2]/1000
            
            ## topoplot
            mask_params = dict(marker='o',
                           markerfacecolor='r',
                           markeredgecolor='k',
                           linewidth=0,
                           markersize=4)
            cur_topo_data = grand_avg['amusics']['familiar'][feature + '_MMN'].copy()
            cur_topo_data.data = ((grand_avg['amusics'][pnm1][feature + '_MMN'].data -
                                   grand_avg['amusics'][pnm2][feature + '_MMN'].data) -
                                  (grand_avg['controls'][pnm1][feature + '_MMN'].data -
                                   grand_avg['controls'][pnm2][feature + '_MMN'].data))
    
            if cluster_t:
               cur_mask = mask_topo[np.argmax(np.abs(cluster_t))]
               
            else: 
               cur_mask = mask_topo.append(np.zeros(grand_avg['amusics']['optimal'][feature + '_MMN'].data.shape,
                                           dtype=bool))
               
            cur_topo_data.plot_topomap(times = lat,
                                       axes = topo_ax,
                                       vmin = -3,
                                       vmax = 3,
                                       colorbar = False,
                                       mask = cur_mask,
                                       mask_params = mask_params,
                                       sensors = False)  
            topo_ax.set_title(feature,fontsize = 14)
            disp_lat = str(int(np.round(lat[0]*1000,0))) + ' ms'
            topo_ax.annotate(disp_lat,xy = (0.5,-0.3),xytext = (0.5,-0.3),
            xycoords = ('axes fraction','axes fraction'),
            textcoords='offset points',
            size=12, ha='center', va='bottom')
            
def simpleFamiliarity(feature,group,ch_names,channs,stat_results,ntests,
                      grand_avg,all_stats,evkd_ax,topo_ax,legend):
    
    ### get stats and plotting masks
    ch_indx = [iidx for iidx,cch in enumerate(ch_names) 
               if (cch in channs)]
    cur_stats = stat_results[feature][group]
    t_vals = cur_stats[0]
    sig_clusters_idx = np.where(cur_stats[2] < .05/ntests)[0]
    masks = [cur_stats[1][x] for x in sig_clusters_idx]
    pvals = cur_stats[2][sig_clusters_idx]
    pval_char = []
    clust_dir = np.zeros(pvals.shape)
    cluster_t = np.zeros(len(masks))
    if len(masks) > 0:
        for idxx,m in enumerate(masks):
            p = pvals[idxx]
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
    elif len(cur_stats[2]) > 0:
#        masks[idxx][:,:] = 0
        p_char = 'p = ' + str(np.round(np.min(cur_stats[2]),decimals = 3))
        pval_char.append([p_char])
    else:
        p_char = 'N.C.F.'
 #       masks[idxx][:,:] = 0
        pval_char.append([p_char])   
        
    ### prepare data
    fam = grand_avg[group]['familiar'][feature + '_MMN'].copy()
    fam = np.mean(fam.pick_channels(channs).data,0)*1e6
    unf = grand_avg[group]['unfamiliar'][feature + '_MMN'].copy()
    unf = np.mean(unf.pick_channels(channs).data,0)*1e6
    diff = fam - unf
    time = grand_avg[group]['familiar'][feature + '_MMN'].times*1000
    
    ## prepare masks with time corrections
    mask_time_idx = np.where((time >= 0) & (time <= 300))[0]

    mask_evkd = []
    mask_topo = []
    
    for idxx, m in enumerate(masks):
        mask_evkd.append(np.zeros(diff.shape[0], dtype=bool))
        mask_topo.append(np.zeros(grand_avg[group]['familiar'][feature + '_MMN'].data.shape,
                             dtype=bool))
    
        mask_evkd[idxx][mask_time_idx] = np.sum(m[:,ch_indx],1) > 0               
        mask_topo[idxx][:,mask_time_idx] = np.transpose(m,(1,0))
        
    ### Get 95% CI
    se_data = (all_stats[group]['familiar'][feature + '_MMN'][:,:,ch_indx] -
               all_stats[group]['unfamiliar'][feature + '_MMN'][:,:,ch_indx])
    
    stdev = np.std(np.mean(se_data,2),0)
    se = stdev/np.sqrt(se_data.shape[0])
    ci_upper = diff + 1.96*1e6*se
    ci_lower = diff - 1.96*1e6*se           
       
    ## plot evoked
    evkd_ax.set_xlim(left = -100, right=400)   
    evkd_ax.set_ylim(bottom=-4, top=4)        
    evkd_ax.hlines(0,xmin = -100,xmax = 400) 
    evkd_ax.vlines(0,ymin = -4,ymax = 4,linestyles = 'solid')        
    
    for isp in range(0,se_data.shape[0]):
        isp_data =  np.mean(se_data[isp,:,:],1)*1e6
        evkd_ax.plot(time,isp_data,'k-',alpha = .05,
                     linewidth = 0.5)
        
    evkd_ax.plot(time,fam,'b--',label = 'familiar')
    evkd_ax.plot(time,unf,'r--',label = 'unfamiliar')
    evkd_ax.plot(time,diff,'k-',label = 'difference')
    evkd_ax.set_xlabel('ms',labelpad = 0) 
    evkd_ax.set_ylabel(r'$\mu$V',labelpad = 0)
    evkd_ax.fill_between(time,ci_lower,ci_upper, color = 'k',alpha = .2)

    # if not mask_evkd:
    #    evkd_ax.annotate('N.C.F',xy = (0,0), xytext = (270,3))
               
    for idxx, em in enumerate(mask_evkd):
        if clust_dir[idxx] == -1:
           color = 'b'
        else:
           color = 'r'
        evkd_ax.fill_between(time[em],-4,4, color = color,alpha = .1)
        
    for idxx, pv in enumerate(pval_char):
        evkd_ax.annotate(pval_char[idxx][0],xy = (0,0), xytext = (200,3-idxx*0.5))      
         
    if legend:
       evkd_ax.legend(fontsize = 8, loc = 2, framealpha = 1,
                      edgecolor = 'black',shadow = True)

                    ## get MMN peak latency
    cur_data = diff.copy()
    peak_idx = ((cur_data[1:-1] - cur_data[2:] < 0) &
              (cur_data[1:-1] - cur_data[0:-2] < 0) &
              (time[1:-1] >= 0) & (time[1:-1] <= 300))
    cur_data = cur_data[1:-1]
    peak_idx2 = np.where((peak_idx == True) &
                        (cur_data == np.amin(cur_data[peak_idx])))
    lat = time[peak_idx2]/1000
    
    ## topoplot
    mask_params = dict(marker='o',
                   markerfacecolor='r',
                   markeredgecolor='k',
                   linewidth=0,
                   markersize=4)
    cur_topo_data = grand_avg[group]['familiar'][feature + '_MMN'].copy()
    cur_topo_data.data = (cur_topo_data.data - 
                          grand_avg[group]['unfamiliar'][feature + '_MMN'].data)
    
    if len(cluster_t) > 0:
       cur_mask = mask_topo[np.argmin(cluster_t)]
       
    else: 
       cur_mask = mask_topo.append(np.zeros(grand_avg[group]['familiar'][feature + '_MMN'].data.shape,
                                   dtype=bool))
       
    cur_topo_data.plot_topomap(times = lat,
                               axes = topo_ax,
                               vmin = -3,
                               vmax = 3,
                               colorbar = False,
                               mask = cur_mask,
                               mask_params = mask_params,
                               sensors = False)  
    topo_ax.set_title(feature,fontsize = 14)
    disp_lat = str(int(np.round(lat[0]*1000,0))) + ' ms'
    topo_ax.annotate(disp_lat,xy = (0.5,-0.3),xytext = (0.5,-0.3),
    xycoords = ('axes fraction','axes fraction'),
    textcoords='offset points', size=12, ha='center', va='bottom')
    
def interFamiliarity(feature,groups,ch_names,channs,stat_results,grand_avg,
                     all_stats,topo_ax,evkd_ax,legend,ntests):
        
    ### get stats and plotting masks
    ch_indx = [iidx for iidx,cch in enumerate(ch_names) 
               if (cch in channs)]
    cur_stats = stat_results[feature]['interaction']
    t_vals = cur_stats[0]
    sig_clusters_idx = np.where(cur_stats[2] < .05/ntests)[0]
    masks = [cur_stats[1][x] for x in sig_clusters_idx]
    pvals = cur_stats[2][sig_clusters_idx]
    pval_char = []
    clust_dir = np.zeros(pvals.shape)
    cluster_t = np.zeros(len(masks))
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
 
    ### prepare data

    amus = (grand_avg['amusics']['familiar'][feature + '_MMN'].data - 
            grand_avg['amusics']['unfamiliar'][feature + '_MMN'].data) 
    amus = np.mean(amus[ch_indx,:],0)*1e6
    cont = (grand_avg['controls']['familiar'][feature + '_MMN'].data -
            grand_avg['controls']['unfamiliar'][feature + '_MMN'].data)
    cont = np.mean(cont[ch_indx,:],0)*1e6
    diff = amus - cont
    time = grand_avg['amusics']['familiar'][feature + '_MMN'].times*1000
    
    ## prepare masks with time corrections
    mask_time_idx = np.where((time >= 0) & (time <= 300))[0]

    mask_evkd = []
    mask_topo = []
    
    for idxx, m in enumerate(masks):
        mask_evkd.append(np.zeros(diff.shape[0], dtype=bool))
        mask_topo.append(np.zeros(grand_avg['amusics']['familiar'][feature + '_MMN'].data.shape,
                             dtype=bool))
    
        mask_evkd[idxx][mask_time_idx] = np.sum(m[:,ch_indx],1) > 0               
        mask_topo[idxx][:,mask_time_idx] = np.transpose(m,(1,0))
        
    ### Get 95% CI
    ci = {}
    for group in groups:
        se_data = (all_stats[group]['familiar'][feature + '_MMN'][:,:,ch_indx] -
                   all_stats[group]['unfamiliar'][feature + '_MMN'][:,:,ch_indx])
    
        stdev = np.std(np.mean(se_data,2),0)
        se = stdev/np.sqrt(se_data.shape[0])
        ci[group] = 1.96*1e6*se   
    
    
    ## plot evoked
    evkd_ax.set_xlim(left = -100, right=400)   
    evkd_ax.set_ylim(bottom=-2, top=2)        
    evkd_ax.hlines(0,xmin = -100,xmax = 400) 
    evkd_ax.vlines(0,ymin = -2,ymax = 2,linestyles = 'solid')        
    evkd_ax.plot(time,amus,'b-',label = 'amusics')
    evkd_ax.plot(time,cont,'r-',label = 'controls')
    evkd_ax.plot(time,diff,'k--',label = 'difference')
    evkd_ax.set_xlabel('ms',labelpad = 0) 
    evkd_ax.set_ylabel(r'$\mu$V',labelpad = 0)
    evkd_ax.fill_between(time,amus + ci['amusics'],
                         amus - ci['amusics'], color = 'b',alpha = .2)
    evkd_ax.fill_between(time,cont + ci['controls'],
                         cont - ci['controls'], color = 'r',alpha = .2)
    
    if not mask_evkd:
       evkd_ax.annotate('N.S.',xy = (0,0), xytext = (100,1.5))
       
    for idxx, em in enumerate(mask_evkd):
        if clust_dir[idxx] == -1:
           color = 'b'
        else:
           color = 'r'
        evkd_ax.fill_between(time[em],-2,2, color = color,alpha = .1)            
        evkd_ax.annotate(pval_char[idxx][0],xy = (0,0), xytext = (270,1.5))
                
    if legend:
       evkd_ax.legend(fontsize = 8, loc = 2, framealpha = 1,
                      edgecolor = 'black',shadow = True)
    
    ## get peak latency of difference
    cur_data = diff.copy()
    peak_idx = ((cur_data[1:-1] - cur_data[2:] < 0) &
              (cur_data[1:-1] - cur_data[0:-2] < 0) &
              (time[1:-1] >= 0) & (time[1:-1] <= 300))
    cur_data = cur_data[1:-1]
    peak_idx2 = np.where((peak_idx == True) &
                        (cur_data == np.amin(cur_data[peak_idx])))
    lat = time[peak_idx2]/1000
    
    ## topoplot
    mask_params = dict(marker='o',
                   markerfacecolor='r',
                   markeredgecolor='k',
                   linewidth=0,
                   markersize=4)
    cur_topo_data = grand_avg['amusics']['familiar'][feature + '_MMN'].copy()
    cur_topo_data.data = ((grand_avg['amusics']['familiar'][feature + '_MMN'].data -
                           grand_avg['amusics']['unfamiliar'][feature + '_MMN'].data) -
                          (grand_avg['controls']['familiar'][feature + '_MMN'].data -
                           grand_avg['controls']['unfamiliar'][feature + '_MMN'].data))

    if len(cluster_t) > 0:
       cur_mask = mask_topo[np.argmin(cluster_t)]
       
    else: 
       cur_mask = mask_topo.append(np.zeros(grand_avg['amusics']['familiar'][feature + '_MMN'].data.shape,
                                   dtype=bool))
       
    cur_topo_data.plot_topomap(times = lat,
                               axes = topo_ax,
                               vmin = -2,
                               vmax = 2,
                               colorbar = False,
                               mask = cur_mask,
                               mask_params = mask_params,
                               sensors = False)  
    topo_ax.set_title(feature,fontsize = 14)
    disp_lat = str(int(np.round(lat[0]*1000,0))) + ' ms'
    topo_ax.annotate(disp_lat,xy = (0.5,-0.3),xytext = (0.5,-0.3),
    xycoords = ('axes fraction','axes fraction'),
    textcoords='offset points',
    size=12, ha='center', va='bottom')