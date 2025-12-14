#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:19:00 2023

@author: ron.teichner
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

ronWideAxons_mapping_path = '/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf__mapping.csv'
ronWideAxons_spikes_path = '/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_conf_0000_spikes_h5_DF.csv'

RonWideAxonsPath = ['/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf_wrt_e_5005_branch_28_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf_wrt_e_5005_branch_26_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf_wrt_e_5005_branch_25_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf_wrt_e_5005_branch_11_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf_wrt_e_5005_bestSnrToas@MA_256.csv',
         ]

paths = ['/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf_wrt_e_5005_branch_28_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf_wrt_e_5005_branch_26_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf_wrt_e_5005_branch_25_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf_wrt_e_5005_branch_11_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/Ron.raw.h5_allConf_wrt_e_5005_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/rec_18680_Jun11_2023/Neuron 21/Trace_20230702_07_46_14.raw.h5__ronAnalysis/Trace_20230702_07_46_14.raw.h5_allConf_wrt_e_5431_branch_58_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/rec_18680_Jun11_2023/Neuron 21/Trace_20230702_07_46_14.raw.h5__ronAnalysis/Trace_20230702_07_46_14.raw.h5_allConf_wrt_e_5431_branch_21_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/rec_18680_Jun11_2023/Neuron 21/Trace_20230629_16_52_14.raw.h5__ronAnalysis/Trace_20230629_16_52_14.raw.h5_allConf_wrt_e_5431_branch_34_bestSnrToas@MA_256.csv',
         '/Users/ron.teichner/Data/MXBIO/Data/recordings/rec_18680_Jun11_2023/Neuron 10/Trace_20230706_07_58_30_spont.raw.h5__ronAnalysis/Trace_20230706_07_58_30_spont.raw.h5_allConf_wrt_e_2108_branch_17_bestSnrToas@MA_256.csv'
         ]

net2 = ['/Users/ron.teichner/Data/MXBIO/Data/recordings/rec_18680_Jun11_2023/Neuron 21/Trace_20230702_07_46_14.raw.h5__ronAnalysis/Trace_20230702_07_46_14.raw.h5_allConf_wrt_e_5431_branch_58_bestSnrToas@MA_256.csv',
'/Users/ron.teichner/Data/MXBIO/Data/recordings/rec_18680_Jun11_2023/Neuron 21/Trace_20230702_07_46_14.raw.h5__ronAnalysis/Trace_20230702_07_46_14.raw.h5_allConf_wrt_e_5431_branch_21_bestSnrToas@MA_256.csv',]

ronWideAxons_mapping = pd.read_csv(ronWideAxons_mapping_path)
ronWideAxons_spikes = pd.read_csv(ronWideAxons_spikes_path)
spikeRefElectrodeNo = 5005
spikeRefChannelNo = ronWideAxons_mapping[ronWideAxons_mapping['electrode'] == spikeRefElectrodeNo]['channel'].to_numpy()[0].astype(int)
spikes_h5_DF_channel = ronWideAxons_spikes[ronWideAxons_spikes['channel']==spikeRefChannelNo]
spikes_h5_DF_channel = spikes_h5_DF_channel.sort_values(by='time')
cellSpikesAmps = spikes_h5_DF_channel['amplitude'].reset_index()
cellSpikesAmpsTimes = spikes_h5_DF_channel['time'].reset_index()
rollingValue = 256
ronWideAxon_amplitudesRolled = cellSpikesAmps.rolling(window=rollingValue, min_periods=1).mean()
ronWideAxon_amplitudesRolled_shiftScale = ronWideAxon_amplitudesRolled - ronWideAxon_amplitudesRolled.mean()
ronWideAxon_amplitudesRolled_shiftScale = 3*ronWideAxon_amplitudesRolled_shiftScale
ronWideAxon_amplitudesRolled_shiftScale = ronWideAxon_amplitudesRolled_shiftScale + 1100

plt.figure(figsize=(6.4*3,8.4))
colors = ['k', 'b', 'g', 'brown', 'r', 'm']

for a,singlePath in enumerate(RonWideAxonsPath):
    print('')
    print('starting ' + singlePath)
    toas = pd.read_csv(singlePath)
    toas.drop(columns=['Unnamed: 0'], inplace=True)
    if a == 0:
        plt.plot(toas['time'], ronWideAxon_amplitudesRolled_shiftScale['amplitude'], color='black', linestyle='dashed', linewidth=1.0, label='amp')
        pd.concat((toas['time'].reset_index(), ronWideAxon_amplitudesRolled.reset_index()), axis=1)[['time','amplitude']].to_csv('/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/neuronAmps.csv')
        pd.concat((toas['time'].reset_index(), cellSpikesAmps.reset_index()), axis=1)[['time','amplitude']].to_csv('/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/neuronAmps_noAVG.csv')
        
            
    for c,column in enumerate(toas.columns[1:]):
        corrWithAmp = pd.Series(toas[column].to_numpy()).corr(pd.Series(ronWideAxon_amplitudesRolled['amplitude'].to_numpy()))
        print(f'(a,c) = {a},{c}: corr with amp = {str(round(corrWithAmp, 2))}')
        if c == 0:
            plt.plot(toas['time'], toas[column]/1e-6, color=colors[a], linewidth=0.5, label=f'Branch {a}')
        else:
            plt.plot(toas['time'], toas[column]/1e-6, color=colors[a], linewidth=0.5)
            
plt.xlabel('sec')
plt.ylabel('us')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(6.4*3,8.4/1.5))
startIdx = np.where(toas['time'] < toas['time'].median() - 3600/2)[0][-1]
stopIdx = np.where(toas['time'] >= toas.loc[startIdx,'time'] + 3600)[0][0]
colors = ['k', 'b', 'g', 'brown', 'r', 'm']
plt.subplot(1,2,1)
for a,singlePath in enumerate(RonWideAxonsPath):
    print('')
    print('starting ' + singlePath)
    toas = pd.read_csv(singlePath)
    toas.drop(columns=['Unnamed: 0'], inplace=True)
    if a == 0:
        plt.plot(toas.loc[startIdx:stopIdx, 'time'], ronWideAxon_amplitudesRolled_shiftScale.loc[startIdx:stopIdx, 'amplitude'], color='black', linestyle='dashed', linewidth=1.0, label='amp')
        pd.concat((toas['time'].reset_index(), ronWideAxon_amplitudesRolled.reset_index()), axis=1)[['time','amplitude']].to_csv('/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/neuronAmps.csv')
        pd.concat((toas['time'].reset_index(), cellSpikesAmps.reset_index()), axis=1)[['time','amplitude']].to_csv('/Users/ron.teichner/Data/MXBIO/Data/recordings/Ron Wide Axons/Ron.raw.h5__ronAnalysis/neuronAmps_noAVG.csv')
        
            
    for c,column in enumerate(toas.columns[1:]):
        corrWithAmp = pd.Series(toas[column].to_numpy()).corr(pd.Series(ronWideAxon_amplitudesRolled['amplitude'].to_numpy()))
        print(f'(a,c) = {a},{c}: corr with amp = {str(round(corrWithAmp, 2))}')
        if c == 0:
            plt.plot(toas.loc[startIdx:stopIdx, 'time'], toas.loc[startIdx:stopIdx, column]/1e-6, color=colors[a], linewidth=0.5, label=f'Branch {a}')
        else:
            plt.plot(toas.loc[startIdx:stopIdx, 'time'], toas.loc[startIdx:stopIdx, column]/1e-6, color=colors[a], linewidth=0.5)
            
plt.xlabel('sec')
plt.ylabel('us')
plt.grid()
plt.legend()

plt.subplot(1,4,3)
for a,singlePath in enumerate(RonWideAxonsPath):
    toas = pd.read_csv(singlePath)
    toas.drop(columns=['Unnamed: 0'], inplace=True)
    for c,column in enumerate(toas.columns[1:]):
        corrWithAmp = pd.Series(toas.loc[startIdx:stopIdx, column].to_numpy()).corr(pd.Series(ronWideAxon_amplitudesRolled.loc[startIdx:stopIdx, 'amplitude'].to_numpy()))
        if a==3 and c==0:
            plt.scatter(x=ronWideAxon_amplitudesRolled.loc[startIdx:stopIdx, 'amplitude'], y=toas.loc[startIdx:stopIdx, column]/1e-6, s=1, label=f'Pearson correlation = {str(round(corrWithAmp, 2))}')
    
plt.xlabel('uVolt')
plt.ylabel('us')
plt.grid()
plt.legend()
plt.show()
    
plt.figure(figsize=(6.4*3,8.4))
colors = ['k', 'b', 'g', 'brown', 'r', 'm']
for a,singlePath in enumerate(net2):
    print('')
    print('starting ' + singlePath)
    toas = pd.read_csv(singlePath)
    toas.drop(columns=['Unnamed: 0'], inplace=True)
            
    for c,column in enumerate(toas.columns[1:]):
        if c == 0:
            plt.plot(toas['time'], toas[column]/1e-6, color=colors[a], linewidth=0.5, label=f'Axon {a}')
        else:
            plt.plot(toas['time'], toas[column]/1e-6, color=colors[a], linewidth=0.5)
            
plt.xlabel('sec')
plt.ylabel('us')
plt.grid()
plt.legend()
plt.show()

    
for singlePath in paths:
    print('')
    print('starting ' + singlePath)
    toas = pd.read_csv(singlePath)
    toas.drop(columns=['Unnamed: 0'], inplace=True)
    '''
    plt.figure()
    plt.plot(toas.index, toas['time'])
    plt.xlabel('events')
    plt.ylabel('sec')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(6.4*3,8.4))
    
    plt.subplot(1,2,1)
    for column in toas.columns[1:]:
        plt.plot(toas['time'], toas[column]/1e-6, label=column)
    plt.xlabel('sec')
    plt.ylabel('us')
    plt.grid()
    plt.legend()
    
    plt.subplot(1,2,2)
    for column in toas.columns[1:]:
        plt.plot(toas.index, toas[column]/1e-6, label=column)
    plt.xlabel('events')
    plt.ylabel('us')
    plt.grid()
    plt.legend()
    
    plt.show()
    '''
    plt.figure(figsize=(6.4*6,8.4*2))
    
    plt.subplot(2,1,1)
    for column in toas.columns[1:]:
        plt.plot(toas['time'], toas[column]/1e-6, label=column)
    plt.xlabel('sec')
    plt.ylabel('us')
    plt.grid()
    plt.legend()
    
    plt.subplot(2,1,2)
    res = 120 # sec
    nEventsList = list()
    tVecList = list()
    for t in toas['time'].to_numpy().tolist():
        nEventsList.append(np.logical_and(toas['time'] >=t-res, toas['time'] < t).sum())
        
    plt.plot(toas['time'], nEventsList)
    plt.xlabel('sec')
    plt.ylabel('nEvents in ' + str(int(res)) + ' sec')
    plt.grid()
    plt.show()
    '''
    res = 30 # sec
    nEventsList = list()
    for t in toas['time'].to_numpy().tolist():
        nEventsList.append(np.logical_and(toas['time'] >=t-res, toas['time'] < t).sum())
    
    for column in toas.columns[1:]:
        plt.figure()
        plt.scatter(x=nEventsList, y=toas[column]/1e-6, label=column)
        plt.xlabel('nEvents in ' + str(int(res)) + ' sec')
        plt.ylabel('us')
        plt.grid()
        plt.legend()
        plt.show()
    '''
    
    
        
    

    
plt.figure()
plt.subplot(1,2,1)
for i,singlePath in enumerate(paths):
    print('')
    print('starting ' + singlePath)
    toas = pd.read_csv(singlePath)
    toas.drop(columns=['Unnamed: 0'], inplace=True)
    plt.plot(toas.mean()[1:]/1e-3, toas.std()[1:]/1e-6, '--o', label=str(i))
plt.xlabel(r'$E[\tau]$ [ms]')
plt.ylabel(r'$\sigma(\tau)$ [us]')
plt.grid()
plt.legend()
plt.tight_layout()

plt.subplot(1,2,2)
for i,singlePath in enumerate(paths):
    print('')
    print('starting ' + singlePath)
    toas = pd.read_csv(singlePath)
    toas.drop(columns=['Unnamed: 0'], inplace=True)
    plt.plot(toas.mean()[1:]/1e-3, toas.std()[1:]/toas.mean()[1:], '--o', label=str(i))
plt.xlabel(r'$E[\tau]$ [ms]')
plt.ylabel(r'$\frac{\sigma(\tau)}{E[\tau]}$')
plt.grid()
plt.legend()
plt.ylim([0.0,0.06])
plt.tight_layout()
plt.show()


Id = -1
Id_toas = -1
Id_vel = -1
for singlePath in paths:
    completeFeatures = ['dT','v_nMinus1','v_n']
    df = pd.DataFrame(columns=["time", "Id", "batch"] + completeFeatures)
    
    completeFeatures_toas = ['dT', 'd_toa_nMinus1', 'd_toa_n', 'd_toa_nPlus1']
    df_toas = pd.DataFrame(columns=["time", "Id", "batch"] + completeFeatures_toas)
    
    completeFeatures_velocities = ['dT', 'v_nMinus1', 'v_nPlus1']
    df_velocities = pd.DataFrame(columns=["time", "Id", "batch"] + completeFeatures_velocities)


    print('')
    print('starting ' + singlePath)
    toas = pd.read_csv(singlePath)
    toas.drop(columns=['Unnamed: 0'], inplace=True)
    dtoas = toas.iloc[:, 1:].diff(axis=1).iloc[:,1:]
    relative_dtoas = dtoas/dtoas.median(axis=0)
    relative_dtoas = pd.concat((toas['time'].diff(), relative_dtoas), axis=1)
    for cIdx in range(1, relative_dtoas.shape[1]-1):
        Id += 1
        r = pd.DataFrame(columns=["time", "Id", "batch"] + completeFeatures, data=np.concatenate((toas['time'].to_numpy()[:,None], Id*np.ones((relative_dtoas.shape[0],1)), np.zeros((relative_dtoas.shape[0],1)), relative_dtoas.iloc[:,[0,cIdx,cIdx+1]].to_numpy()), axis=1))
        df = pd.concat((df, r), axis=0)
    
    original_toas = toas.iloc[:,2:]     # skipping first electrode     
    original_toas = pd.concat((toas['time'].diff(), original_toas), axis=1)
    for cIdx in range(1, original_toas.shape[1]-2):
        Id_vel += 1
        tau_nMinus1, tau_n, tau_nPlus1 = original_toas.iloc[:,cIdx], original_toas.iloc[:,cIdx+1], original_toas.iloc[:,cIdx+2]
        v_nMinus1 = (tau_nMinus1.median()/tau_nMinus1).to_numpy()
        v_nPlus1 = ((tau_nPlus1.median() - tau_n.median())/(tau_nPlus1 - tau_n)).to_numpy()
        r = pd.DataFrame(columns=["time", "Id", "batch"] + completeFeatures_velocities, data=np.concatenate((toas['time'].to_numpy()[:,None], Id_toas*np.ones((original_toas.shape[0],1)), np.zeros((original_toas.shape[0],1)), original_toas.iloc[:,0].to_numpy()[:,None], v_nMinus1[:,None], v_nPlus1[:,None]), axis=1))
        df_velocities = pd.concat((df_velocities, r), axis=0)
    
    relative_toas = toas.iloc[:,2:] # skipping first electrode     
    relative_toas = relative_toas/relative_toas.median(axis=0)
    relative_toas = pd.concat((toas['time'].diff(), relative_toas), axis=1)
    for cIdx in range(1, relative_toas.shape[1]-2):
        Id_toas += 1
        r = pd.DataFrame(columns=["time", "Id", "batch"] + completeFeatures_toas, data=np.concatenate((toas['time'].to_numpy()[:,None], Id_toas*np.ones((relative_toas.shape[0],1)), np.zeros((relative_toas.shape[0],1)), relative_toas.iloc[:,[0,cIdx,cIdx+1,cIdx+2]].to_numpy()), axis=1))
        df_toas = pd.concat((df_toas, r), axis=0)

    df.dropna(axis=0, inplace=True)
    df.to_csv('/Users/ron.teichner/Data/MXBIO/Data/recordings/relativeVelocities.csv')
    
    
    #df_toas.iloc[:,4:][df_toas.iloc[:,4:] > df_toas.iloc[:,4:].quantile(0.95).min()] = np.nan
    #df_toas.iloc[:,4:][df_toas.iloc[:,4:] < df_toas.iloc[:,4:].quantile(0.05).min()] = np.nan
    df_toas.dropna(axis=0, inplace=True)
    df_toas.to_csv('/Users/ron.teichner/Data/MXBIO/Data/recordings/relativeToas.csv')
    
    df_velocities.dropna(axis=0, inplace=True)
    
    
    d_toa_nMinus1, d_toa_n, d_toa_nPlus1 = df_toas['d_toa_nMinus1'], df_toas['d_toa_n'], df_toas['d_toa_nPlus1']
    
    plt.figure()
    plt.scatter(x=df_velocities['v_nMinus1'], y=df_velocities['v_nPlus1'], s=1)
    corr = df_velocities['v_nMinus1'].corr(df_velocities['v_nPlus1'])
    plt.xlabel(r'$\bar{v}^{n-1}$')
    plt.ylabel(r'$\bar{v}^{n+1}$')
    plt.grid()
    plt.title(f'corr = {str(round(corr, 3))}')
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(x=d_toa_nMinus1, y=d_toa_n - d_toa_nMinus1, s=1)
    corr = d_toa_nMinus1.corr(d_toa_n-d_toa_nMinus1)
    plt.xlabel(r'$\bar{\tau}^{n-1}$')
    plt.ylabel(r'$\bar{\tau}^{n} - \bar{\tau}^{n-1}$')
    plt.grid()
    plt.title(f'corr = {str(round(corr, 3))}')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    plt.subplot(1,2,2)
    plt.scatter(x=d_toa_nMinus1, y=d_toa_n, s=1)
    corr = d_toa_nMinus1.corr(d_toa_n)
    plt.xlabel(r'$\bar{\tau}^{n-1}$')
    plt.ylabel(r'$\bar{\tau}^{n}$')
    plt.grid()
    plt.title(f'corr = {str(round(corr, 3))}')
    ax2 = plt.gca()
    ax2.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
            
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(x=d_toa_nMinus1, y=d_toa_nPlus1, s=1)
    corr = d_toa_nMinus1.corr(d_toa_nPlus1)
    plt.xlabel(r'$\bar{\tau}^{n-1}$')
    plt.ylabel(r'$\bar{\tau}^{n+1}$')
    plt.grid()
    plt.title(f'corr = {str(round(corr, 3))}')
    plt.tight_layout()
    
    plt.subplot(1,2,2)
    plt.scatter(x=d_toa_nMinus1, y=d_toa_n, s=1)
    corr = d_toa_nMinus1.corr(d_toa_n)
    plt.xlabel(r'$\bar{\tau}^{n-1}$')
    plt.ylabel(r'$\bar{\tau}^{n}$')
    plt.grid()
    plt.title(f'corr = {str(round(corr, 3))}')
    plt.tight_layout()
    plt.show()
    
    
    plt.figure()
    #plt.subplot(1,2,1)
    plt.scatter(x=d_toa_nMinus1, y=d_toa_nPlus1, s=1, label=r'$\bar{\tau}^{n+1}$' + r' vs $\bar{\tau}^{n-1}$' + f'corr = {str(round(corr, 3))}')
    corr = d_toa_nMinus1.corr(d_toa_nPlus1)
    plt.xlabel(r'$\bar{\tau}^{n-1}$')
    plt.ylabel(r'$\bar{\tau}^{n+1}$')
    plt.grid()
    #plt.title(f'corr = {str(round(corr, 3))}')
    plt.tight_layout()
    
    #plt.subplot(1,2,2)
    plt.scatter(x=d_toa_nMinus1, y=d_toa_n, s=1, label=r'$\bar{\tau}^{n}$' + r' vs $\bar{\tau}^{n-1}$' + f'corr = {str(round(corr, 3))}')
    corr = d_toa_nMinus1.corr(d_toa_n)
    plt.xlabel(r'$\bar{\tau}^{n-1}$')
    plt.ylabel(r'$\bar{\tau}^{n}$')
    plt.grid()
    #plt.title(f'corr = {str(round(corr, 3))}')
    plt.tight_layout()
    plt.legend()
    plt.show()
