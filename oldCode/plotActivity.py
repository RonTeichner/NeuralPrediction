import json
import sys
import numpy as np
import torch
import scipy.io as sio
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy import interpolate
import h5py
import os
import datetime
import re
import platform
#from joblib import Parallel, delayed
import multiprocessing
from read_raw_func import *
 
plt.close('all')
axx,ab = 16/3,9/3

if platform.system() == 'Linux':
    path2Dropbox = '/media/ront/4d9d1135-ebb8-4cdb-b6b8-81eb1dd36cb1/media/ront/MXBIO/'#'/media/ront/4d9d1135-ebb8-4cdb-b6b8-81eb1dd36cb1/media/ront/DropboxTechnionBackup/'
else:
    path2Dropbox = '/Users/ron.teichner/Data/MXBIO/'


# read file
#print(f'inputs from keyboard are {sys.argv}')
if len(sys.argv) > 1:
    jsonFileName = sys.argv[1]
else:
    jsonFileName = './runExp.json'
print(f'json file name is {jsonFileName}')

with open(jsonFileName, 'r') as myfile:
    runCommands=json.loads(myfile.read())['config'][0]
    
path2ExpWithinDropbox = runCommands["path2ExpWithinDropbox"]
nameOf_h5 = runCommands["nameOf_h5"]
path2metrics_data_xlsxWithinDropbox = runCommands["path2metrics_data_xlsxWithinDropbox"]
nConfigurations = runCommands["nConfigurations"]

if 'onlyVisit' in runCommands.keys():
    onlyVisit = runCommands['onlyVisit'] == 1
else:
    onlyVisit = False
    
if 'learnActivity' in runCommands.keys():
    learnActivity = runCommands['learnActivity'] == 1
else:
    learnActivity = False
    
if 'enableLearnActivityParallelRun' in runCommands.keys():
    enableLearnActivityParallelRun = runCommands['enableLearnActivityParallelRun'] == 1
else:
    enableLearnActivityParallelRun = False

use_hpfAvg = runCommands["use_hpfAvg"]
process_hpf = runCommands["process_hpf"]
processWindow = runCommands["processWindow"]

processType = runCommands["processType"]
processElectrodes = runCommands["processElectrodes"]

if 'specificProcessedElectrodeList' in runCommands.keys():
    specificProcessedElectrodeList = runCommands['specificProcessedElectrodeList']
else:
    specificProcessedElectrodeList = []

if 'stimuliElectrodeNo' in runCommands.keys():
    stimuliElectrodeNo = runCommands["stimuliElectrodeNo"]
else:
    stimuliElectrodeNo = -1

if 'Neuron' in runCommands.keys():
    Neuron = runCommands["Neuron"]
else:
    Neuron = -1

if 'removeStimuliInterference' in runCommands.keys():
    removeStimuliInterference = runCommands["removeStimuliInterference"]==1
else:
    removeStimuliInterference = False

if 'cropStartingAt' in runCommands.keys():
    cropStartingAt = runCommands['cropStartingAt']
else:
    cropStartingAt = 0e-3
    
if 'enablePlotPerElectrode' in runCommands.keys():
    enablePlotPerElectrode = runCommands['enablePlotPerElectrode'] == 1
else:
    enablePlotPerElectrode = False


#onlyVisit = False
#learnActivity = False
#enableLearnActivityParallelRun = False # after parallel run must do a regular run
#enablePlotPerElectrode = True
#cropStartingAt = 0e-3 # sec
enableElectrodeInvestigation = False
investigationElectrode = []
#use_hpfAvg, process_hpf, processWindow = True, True, 30e-3
#processType = 'neuronIsRef' # {'neuronIsRef', stimuliElectrodeIsRef} 
##stimuliElectrodeNo = 14457
#Neuron = 21
#processElectrodes = 'all' #{'registered2Neuron', 'all','specific'}
#specificProcessedElectrodeList = []
#path2ExpWithinDropbox = 'Data/recordings/rec_18680_Jun11_2023/'
path2exp = path2Dropbox + path2ExpWithinDropbox
#nameOf_h5 = 'Trace_20230629_16_52_14.raw.h5'
filename = path2exp + nameOf_h5
#path2metrics_data_xlsxWithinDropbox = 'Metrics/18680_Jun11_2023_metrics/metrics_data.xlsx'
path2metrics_data_xlsx = path2Dropbox + path2metrics_data_xlsxWithinDropbox
#nConfigurations = 1
configurations = [str(i).zfill(4) for i in range(nConfigurations)]

ListOfGoodSNR_electrodesForPlot = []#list(set([11376, 11374, 8738, 1270, 14465, 13141, 8056, 8711, 13569, 8049, 11812, 14898, 12915, 13788, 11372, 14455, 8493, 14681, 14020, 15556, 13794, 15118, 14463, 14464, 11818, 7830, 11379, 11381, 15337]))

if 'analysisName' in runCommands.keys():
    analysisName = runCommands["analysisName"]
else:
    analysisName = ''
analysisLibrary = path2exp + nameOf_h5 + '_' + analysisName + '_ronAnalysis/'


with h5py.File(filename,'r') as hf:
    hf.visit(print)

if onlyVisit:
    exit()
    

if not(os.path.isdir(analysisLibrary)):
    os.mkdir(analysisLibrary)


if learnActivity:
    for configuration in configurations:        
        settingPointers = ['/data_store/data' + configuration + '/settings/gain', '/data_store/data' + configuration + '/settings/hpf', '/data_store/data' + configuration + '/settings/lsb', '/data_store/data' + configuration + '/settings/sampling', '/data_store/data' + configuration + '/settings/spike_threshold']
        with h5py.File(filename, mode='r') as h5f:
            rawPointer = settingPointers[0]
            gain = np.array(h5f[rawPointer][:])[0]
            print(f'The gain is {gain}')
            
            rawPointer = settingPointers[1]
            hpf = np.array(h5f[rawPointer][:])[0]
            print(f'The hpf is {hpf}')
            
            rawPointer = settingPointers[2]
            lsb = np.array(h5f[rawPointer][:])[0]/1000 # volt
            print(f'The lsb is {lsb}')
            
            rawPointer = settingPointers[3]
            fs = np.array(h5f[rawPointer][:])[0] # hz
            print(f'The fs is {fs}')
            
            rawPointer = settingPointers[4]
            spikeDetectionFactor = np.array(h5f[rawPointer][:])[0]
            print(f'The spike detection factor is {spikeDetectionFactor}')
        
        
        print('')
        rawPointers = ['/data_store/data' + configuration + '/groups/routed/trigger_channels']
        with h5py.File(filename, mode='r') as h5f:
            for rawPointer in rawPointers:
                if rawPointer in h5f:
                    rawData = np.array(h5f[rawPointer][:])
                    print(rawPointer + f' is of shape {rawData.shape}; Trigger channels are {rawData}')
                    triggerChannels = rawData
                else:
                    triggerChannels = None
        
        
        
        print('')
        rawPointers = ['/data_store/data' + configuration + '/groups/routed/frame_nos']
        with h5py.File(filename, mode='r') as h5f:
            for rawPointer in rawPointers:
                rawData = np.array(h5f[rawPointer][:])
                print(rawPointer + f' is of shape {rawData.shape} - frame numbers from which data was recorded')
                tVec = rawData/fs # sec
        
        print('')
        rawPointers = ['/data_store/data' + configuration + '/spikes']
        path2spikeFile = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_spikes_h5_DF.csv'
        if os.path.isfile(path2spikeFile):
            spikes_h5_DF = pd.read_csv(path2spikeFile)
        else:
            with h5py.File(filename, mode='r') as h5f:
                for rawPointer in rawPointers:
                    rawData = np.array(h5f[rawPointer][:])
                    print(rawPointer + f' is of shape {rawData.shape} and represents all spikes metadata')
                    '''
                    if torch.cuda.is_available():
                        spikes_h5_DF = pd.DataFrame(data=np.zeros((rawData.shape[0], 3)), columns=['time', 'channel', 'amplitude'])
                        for i in range(rawData.shape[0]):
                            if np.mod(i,1000)==0: 
                                print(f'spikes_h5_DF: {i} out of {rawData.shape[0]}')
                            spikes_h5_DF.loc[len(spikes_h5_DF.index)] = np.asarray([rawData[i][0]/fs, rawData[i][1], rawData[i][2]])
                    else:
                    '''
                    spikes_h5_DF = pd.DataFrame(data=rawData, columns=['time', 'channel', 'amplitude'])
                    spikes_h5_DF.loc[:,'time'] = np.asarray([t[0]/fs for t in list(rawData)])
                    
                spikes_h5_DF.to_csv(path2spikeFile)    
        
        rawPointers = ['/data_store/data' + configuration + '/groups/routed/channels']
        with h5py.File(filename, mode='r') as h5f:
            channels = np.array(h5f[rawPointers[0]][:])
        
        print('')
        mapping = '/data_store/data' + configuration + '/settings/mapping'
        with h5py.File(filename, mode='r') as h5f:
            rawPointer = mapping
            rawData = np.array(h5f[rawPointer][:])
            if not(triggerChannels is None):
                mappingDF = pd.DataFrame(columns=['channel', 'electrode', 'x', 'y', 'isTrigger'])
                for i in range(rawData.shape[0]):
                    mappingDF.loc[len(mappingDF.index)] = np.asarray([rawData[i][0], rawData[i][1], rawData[i][2], rawData[i][3], rawData[i][0] in triggerChannels])
                print(rawPointer + f' is of shape {rawData.shape} representing the mapping of {rawData.shape} electrodes')
                print(rawPointer + f'[0] is of len {len(rawData[0])} representing the mapping of channel {rawData[0][0]}, electrode {rawData[0][1]} which is located at (x,y) = {rawData[0][2]},{rawData[0][3]}')
            else:
                mappingDF = pd.DataFrame(columns=['channel', 'electrode', 'x', 'y'])
                for i in range(rawData.shape[0]):
                    mappingDF.loc[len(mappingDF.index)] = np.asarray([rawData[i][0], rawData[i][1], rawData[i][2], rawData[i][3]])
                print(rawPointer + f' is of shape {rawData.shape} representing the mapping of {rawData.shape} electrodes')
                print(rawPointer + f'[0] is of len {len(rawData[0])} representing the mapping of channel {rawData[0][0]}, electrode {rawData[0][1]} which is located at (x,y) = {rawData[0][2]},{rawData[0][3]}')
            nSpikes, processedElectrodeIndexInRawRecordings = np.zeros(mappingDF.shape[0]), np.zeros(mappingDF.shape[0])
            for i,channel in enumerate(mappingDF['channel']):
                nSpikes[i] = int((spikes_h5_DF['channel'] == channel).sum())
                processedElectrodeIndexInRawRecordings[i] = np.where(channels == channel)[0][0]
            mappingDF.insert(loc=mappingDF.shape[1], column='nSpikes', value=nSpikes)
            mappingDF.insert(loc=mappingDF.shape[1], column='idxInRawData', value=processedElectrodeIndexInRawRecordings)
            
            mappingDF.insert(loc=mappingDF.shape[1], column='branch', value=-np.ones(mappingDF.shape[0]))
            if not(path2metrics_data_xlsx is None):
                metrics_data_Neuron = pd.read_excel(path2metrics_data_xlsx, sheet_name='Axon - Neuron Level')
                metrics_data_Axon = pd.read_excel(path2metrics_data_xlsx, sheet_name='Axon - Tracking Info')
                metrics_data_Axon.insert(loc=0, column='electrode', value=-1)
                metrics_data_Axon.insert(loc=0, column='closest mapped electrode y', value=-1)
                metrics_data_Axon.insert(loc=0, column='closest mapped electrode x', value=-1)
                metrics_data_Axon.insert(loc=0, column='closest mapped electrode dist', value=-1)
            
                for i in metrics_data_Axon.index:
                    x, y = metrics_data_Axon.loc[i,'x Position [µm]'], metrics_data_Axon.loc[i,'y Position [µm]']
                    distances = np.sqrt(np.power(mappingDF['x']-x, 2) + np.power(mappingDF['y']-y, 2))
                    closestElectrodeIdx = np.argmin(distances)
                    
                    metrics_data_Axon.loc[i, 'closest mapped electrode x'] = mappingDF['x'][closestElectrodeIdx]
                    metrics_data_Axon.loc[i, 'closest mapped electrode y'] = mappingDF['y'][closestElectrodeIdx]
                    metrics_data_Axon.loc[i, 'closest mapped electrode dist'] = distances[closestElectrodeIdx]
                    
                    electrode = mappingDF[np.logical_and(mappingDF['x'] == x, mappingDF['y'] == y)]['electrode']
                    if electrode.shape[0] > 0:
                        metrics_data_Axon.loc[i, 'electrode'] = electrode.to_numpy()[0].astype(int)
                        assert distances[closestElectrodeIdx] == 0
                        #print(f'distance for electrode {electrode.to_numpy()[0].astype(int)} between metrics_data_Axon and mappingDF is {distances[closestElectrodeIdx]}')
                        
                NeuronElectrodeNo = metrics_data_Neuron[metrics_data_Neuron['neuron']==Neuron]['Electrode Number'].to_numpy()[0]  
                branchs = metrics_data_Axon[metrics_data_Axon['neuron'] == Neuron]['branch'].unique().tolist()
                electrodes_on_branch_list = list()
                for branch in branchs:
                    electrodes_on_branch = metrics_data_Axon[np.logical_and(metrics_data_Axon['neuron'] == Neuron, metrics_data_Axon['branch'] == branch)]['electrode']
                    electrodes_on_branch = electrodes_on_branch[electrodes_on_branch > 0].to_numpy().tolist()
                    mappingDF.loc[mappingDF['electrode'].isin(electrodes_on_branch), 'branch'] = branch
                    electrodes_on_branch_list.append(electrodes_on_branch)
                                
            mappingDF.to_csv(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_mapping.csv')
            
        
        
        if processType == 'stimuliElectrodeIsRef':
            spikeRefElectrodeNo = mappingDF[mappingDF['electrode'] == stimuliElectrodeNo]['electrode'].to_numpy()[0].astype(int)
        elif processType == 'neuronIsRef':    
            spikeRefElectrodeNo = metrics_data_Neuron[metrics_data_Neuron['neuron']==Neuron]['Electrode Number'].to_numpy()[0]
        
        if processElectrodes == 'registered2Neuron':
            electrodeProcessList = [e for e in metrics_data_Axon[metrics_data_Axon['neuron'] == Neuron]['electrode'] if e > -1]
        elif processElectrodes == 'all':    
            electrodeProcessList = mappingDF[np.logical_not(mappingDF['electrode'] == spikeRefElectrodeNo)]['electrode'].tolist()
        elif processElectrodes == 'specific':    
            electrodeProcessList = specificProcessedElectrodeList

        electrodeProcessList = [spikeRefElectrodeNo] + list(set(electrodeProcessList))
        electrodeProcessList = [int(r) for r in electrodeProcessList]
        print(f'{len(electrodeProcessList)} electrodes to process')
        
        timeShiftVsRef = 0
        path2my_avgPatternDf = analysisLibrary + nameOf_h5 + '_conf_' + configuration + 'wrt_e_' + str(spikeRefElectrodeNo) + '_avgPatternsDF.csv'
        if os.path.isfile(path2my_avgPatternDf):
            avgPatternDf = pd.read_csv(path2my_avgPatternDf)
            partialFile = True
        else:
            partialFile = False
        
        if learnActivity:
            
            
            ei, processedElectrodeNo = 0, electrodeProcessList[0]
            cstr = 'e ' + str(processedElectrodeNo) + ' hpf'
            Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
            path2my_mySpikeDf_withSamples = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithSamples_DF.csv'
            if not(os.path.isfile(path2my_mySpikeDf_withSamples)):
                timeSeriesDf = createTimeSeriesDf(filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, tVec, fs)
            else:
                timeSeriesDf = None
            print(f'{datetime.datetime.now()}: starting processElectrode with spikeRef = {spikeRefElectrodeNo} and processed = {processedElectrodeNo}; electrode {ei} out of {len(electrodeProcessList)}')
            _, mySpikeDf = processElectrode(None, filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, timeShiftVsRef, timeSeriesDf, spikes_h5_DF, fs, False, True, Id_str, use_hpfAvg, process_hpf, processWindow, removeStimuliInterference, stimuliElectrodeNo)
            
            activityLoopInputList = list()
            if enableLearnActivityParallelRun:
                for ei, processedElectrodeNo in enumerate(np.random.permutation(electrodeProcessList).tolist()):
                    activityLoopInputList.append((ei, len(electrodeProcessList), filename, analysisLibrary, nameOf_h5, mySpikeDf, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, tVec, fs, use_hpfAvg, process_hpf, processWindow, spikes_h5_DF, timeShiftVsRef, removeStimuliInterference, stimuliElectrodeNo))
            else:
                for ei, processedElectrodeNo in enumerate(electrodeProcessList):
                    activityLoopInputList.append((ei, len(electrodeProcessList), filename, analysisLibrary, nameOf_h5, mySpikeDf, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, tVec, fs, use_hpfAvg, process_hpf, processWindow, spikes_h5_DF, timeShiftVsRef, removeStimuliInterference, stimuliElectrodeNo))
            
            #pool_obj = multiprocessing.Pool()
            #ans = pool_obj.map(activityLoop, activityLoopInputList)


            for ei in range(len(electrodeProcessList)):   
                processedElectrodeNo = activityLoopInputList[ei][9]
                cstr = 'e ' + str(processedElectrodeNo) + ' hpf'
                if enableLearnActivityParallelRun:
                    activityLoop(*activityLoopInputList[ei])
                else:
                    avgPatternSeries = activityLoop(*activityLoopInputList[ei])
                    if ei == 0:
                        series_tVec = np.arange(len(avgPatternSeries))/fs
                        avgPatternDf = pd.DataFrame(columns=['time', 'e ' + str(processedElectrodeNo) + ' hpf'], data=np.concatenate((series_tVec[:,None], avgPatternSeries.to_numpy()[:,None]), axis=1))
                    else:
                        df = pd.DataFrame(columns=[cstr], data=avgPatternSeries.to_numpy()[:,None])
                        avgPatternDf = pd.concat([avgPatternDf, df], axis=1)
                        avgPatternDf.to_csv(path2my_avgPatternDf)
            if not enableLearnActivityParallelRun:    
                avgPatternDf.to_csv(path2my_avgPatternDf)

if not enableLearnActivityParallelRun:    
    # union all mappingDF from all configurations:
    path2union_mappingDF = analysisLibrary + nameOf_h5 + '_allConf_' + '_mapping.csv'    
    if not(os.path.isfile(path2union_mappingDF)):
        for i, configuration in enumerate(configurations):
            mappingDF = pd.read_csv(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_mapping.csv')
            mappingDF = mappingDF.loc[:,[np.logical_not(column.find('Unnamed:') >= 0) for column in mappingDF.columns]]
            mappingDF.insert(loc=0, column='config', value=int(configuration))
            if i == 0:
                union_mappingDF = mappingDF
            else:
                union_mappingDF = pd.concat((union_mappingDF, mappingDF[np.logical_not(mappingDF['electrode'].isin(union_mappingDF['electrode'].to_list()))]), axis=0, ignore_index=True)
        union_mappingDF.to_csv(path2union_mappingDF)    
    else:
        union_mappingDF = pd.read_csv(path2union_mappingDF)
    
    
    if not(path2metrics_data_xlsx is None):
        metrics_data_Neuron = pd.read_excel(path2metrics_data_xlsx, sheet_name='Axon - Neuron Level')
        metrics_data_Axon = pd.read_excel(path2metrics_data_xlsx, sheet_name='Axon - Tracking Info')
        metrics_data_Axon.insert(loc=0, column='electrode', value=-1)
        metrics_data_Axon.insert(loc=0, column='closest mapped electrode y', value=-1)
        metrics_data_Axon.insert(loc=0, column='closest mapped electrode x', value=-1)
        metrics_data_Axon.insert(loc=0, column='closest mapped electrode dist', value=-1)
    
        for i in metrics_data_Axon.index:
            x, y = metrics_data_Axon.loc[i,'x Position [µm]'], metrics_data_Axon.loc[i,'y Position [µm]']
            distances = np.sqrt(np.power(union_mappingDF['x']-x, 2) + np.power(union_mappingDF['y']-y, 2))
            closestElectrodeIdx = np.argmin(distances)
            
            metrics_data_Axon.loc[i, 'closest mapped electrode x'] = union_mappingDF['x'][closestElectrodeIdx]
            metrics_data_Axon.loc[i, 'closest mapped electrode y'] = union_mappingDF['y'][closestElectrodeIdx]
            metrics_data_Axon.loc[i, 'closest mapped electrode dist'] = distances[closestElectrodeIdx]
            
            electrode = union_mappingDF[np.logical_and(union_mappingDF['x'] == x, union_mappingDF['y'] == y)]['electrode']
            if electrode.shape[0] > 0:
                metrics_data_Axon.loc[i, 'electrode'] = electrode.to_numpy()[0].astype(int)
                assert distances[closestElectrodeIdx] == 0
                #print(f'distance for electrode {electrode.to_numpy()[0].astype(int)} between metrics_data_Axon and union_mappingDF is {distances[closestElectrodeIdx]}')
        
    if processType == 'stimuliElectrodeIsRef':
        spikeRefElectrodeNo = union_mappingDF[union_mappingDF['electrode'] == stimuliElectrodeNo]['electrode'].to_numpy()[0].astype(int)
    elif processType == 'neuronIsRef':    
        spikeRefElectrodeNo = metrics_data_Neuron[metrics_data_Neuron['neuron']==Neuron]['Electrode Number'].to_numpy()[0]
    
    
    # union all avgPatternDf from all configurations:
    path2my_unionAvgPatternDf = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_avgPatternsDF.csv'     
    if not(os.path.isfile(path2my_unionAvgPatternDf)):
        for i, configuration in enumerate(configurations):
            path2my_avgPatternDf = analysisLibrary + nameOf_h5 + '_conf_' + configuration + 'wrt_e_' + str(spikeRefElectrodeNo) + '_avgPatternsDF.csv'     
            avgPatternDf = pd.read_csv(path2my_avgPatternDf)
            avgPatternDf = avgPatternDf.loc[:, [np.logical_not(column.find('Unnamed:') >= 0) for column in avgPatternDf.columns]]
            if i == 0:
                unionAvgPatternDf = avgPatternDf
            else:
                unionAvgPatternDf = pd.concat((unionAvgPatternDf, avgPatternDf[set(avgPatternDf.columns.to_list()).difference(set(unionAvgPatternDf.columns.to_list()))]), axis=1)
        unionAvgPatternDf.to_csv(path2my_unionAvgPatternDf)
    else:
        unionAvgPatternDf = pd.read_csv(path2my_unionAvgPatternDf)
        
        
    unionAvgPatternDf = unionAvgPatternDf.loc[:, [np.logical_not(column.find('Unnamed:') >= 0) for column in unionAvgPatternDf.columns]]
    #electrodesIn_avgPattern = list(set([int(''.join(x for x in r if x.isdigit())) for r in unionAvgPatternDf.columns.tolist()[1:]]))
    electrodesIn_avgPattern = list(set([numberOutOfStr(r) for r in unionAvgPatternDf.columns.tolist()[1:]]))
    mappingDFOfElectrodesIn_unionAvgPatternDf = union_mappingDF[union_mappingDF['electrode'].isin(electrodesIn_avgPattern)]
    mvAvg_values = list(set([fn[fn.find('_movingAvg_')+11:fn.find('_corrsOptimal')] for fn in os.listdir(analysisLibrary) if '_movingAvg_' in fn and not('_spatial_' in fn)]))
    placeStrs = ['(' + str(int(r)) + ',' + str(int(c)) + ')' for r in [-1,0,1] for c in [-1,0,1] if not(r==0 and c==0)]
    noise_mean, noisePower, power, toa, snr, toa_std, maxCorrVal, maxCorrDTOA, maxCorrElectrode_xDiff, maxCorrElectrode_yDiff = np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0])), np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0])), np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0])), np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0])), np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0])), np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0], len(mvAvg_values))), np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0], len(mvAvg_values))),  np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0], len(mvAvg_values))), np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0], len(mvAvg_values))), np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0], len(mvAvg_values)))
    print(f'{mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0]} electrodes in union avg pattern')
    for i, electrode in enumerate(mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].tolist()):
        configuration = configurations[0]
        path2my_mySpikeDf_withFeaturesSingleRow = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(electrode)) + '_mySpikesWithFeaturesSingleRow_DF.csv'
        if os.path.isfile(path2my_mySpikeDf_withFeaturesSingleRow):
            _mySpikesAll_DF = pd.read_csv(path2my_mySpikeDf_withFeaturesSingleRow)
            _mySpikesAll_DF = pd.DataFrame(columns=_mySpikesAll_DF.iloc[:,0].tolist(), data=_mySpikesAll_DF.iloc[:,1].to_numpy()[None,:])
            print(f'uploaded file ...{str(int(electrode))}' + '_mySpikesWithFeaturesSingleRow_DF.csv:' + f'{i} out of {len(toa)}')
            toa_std_columns = ['toa_std@MA ' + str(int(mvAvg_value)) + ' @e ' + str(int(electrode)) for mvAvg_value in mvAvg_values]
            for t,toa_std_column in enumerate(toa_std_columns):
                if toa_std_column in _mySpikesAll_DF.columns:
                    toa_std[i,t] = _mySpikesAll_DF[toa_std_column].to_numpy()
                else:
                    toa_std[i,t] = np.nan
            #toa_std@MA 128 @e 9432
            maxCorrVal[i] = np.nan*np.ones(len(mvAvg_values))
            maxCorrDTOA[i] = np.nan*np.ones(len(mvAvg_values))
            maxCorrElectrode_xDiff[i] = np.nan*np.ones(len(mvAvg_values))
            maxCorrElectrode_yDiff[i] = np.nan*np.ones(len(mvAvg_values))
            for m,mvAvg_value in enumerate(mvAvg_values):
                placeStr_columns = [placeStr for placeStr in placeStrs]
                spatialCorrRef_column = 'spatialCORR@MA ' + str(int(mvAvg_value)) + ' @e ' + str(int(electrode)) + ' with ' + '(0,0)'
                spatialCorr_columns = ['spatialCORR@MA ' + str(int(mvAvg_value)) + ' @e ' + str(int(electrode)) + ' with ' + placeStr for placeStr in placeStrs]
                spatialDTOA_columns = ['spatialDTOA@MA ' + str(int(mvAvg_value)) + ' @e ' + str(int(electrode)) + ' with ' + placeStr for placeStr in placeStrs]
                # spatialCORR@MA 256 @e 9432 with (-1,0)
                spatialCorrValList, spatialCorrDTOAList, placeStr_columnList = list(), list(), list()
                for spatialCorr_column, spatialDTOA_column, placeStr_column in zip(spatialCorr_columns, spatialDTOA_columns, placeStr_columns):
                    if spatialCorr_column in _mySpikesAll_DF.columns:                    
                        spatialCorrValList.append(_mySpikesAll_DF[spatialCorr_column])
                        spatialCorrDTOAList.append(_mySpikesAll_DF[spatialDTOA_column])
                        placeStr_columnList.append(placeStr_column)
                if len(spatialCorrValList) > 0:
                    maxCorrIdx = np.nanargmax(np.asarray(spatialCorrValList))
                    maxCorrVal[i,m] = spatialCorrValList[maxCorrIdx]
                    maxCorrDTOA[i,m] = spatialCorrDTOAList[maxCorrIdx]
                    maxCorrElectrode_xDiff[i,m] = int(placeStr_columnList[maxCorrIdx][1:placeStr_columnList[maxCorrIdx].find(',')])
                    maxCorrElectrode_yDiff[i,m] = int(placeStr_columnList[maxCorrIdx][placeStr_columnList[maxCorrIdx].find(',')+1:-1])
        else:
            print(f'{path2my_mySpikeDf_withFeaturesSingleRow} file not found')
            toa_std[i] = np.nan*np.ones(len(mvAvg_values))
            maxCorrVal[i] = np.nan*np.ones(len(mvAvg_values))
            maxCorrDTOA[i] = np.nan*np.ones(len(mvAvg_values))
            maxCorrElectrode_xDiff[i] = np.nan*np.ones(len(mvAvg_values))
            maxCorrElectrode_yDiff[i] = np.nan*np.ones(len(mvAvg_values))
        noise_mean[i] = unionAvgPatternDf['e ' + str(int(electrode)) + ' hpf'].median()
        noisePower[i] = np.power(unionAvgPatternDf['e ' + str(int(electrode)) + ' hpf'] - noise_mean[i], 2).median()
        if noisePower[i] == 0:
            noisePower[i] = np.nan
        #noise = unionAvgPatternDf[unionAvgPatternDf['time'] > processWindow - 4e-3]['e ' + str(int(electrode)) + ' hpf']
        #noisePower = noise.var()  # per sample
        if electrode == spikeRefElectrodeNo:
            sig = unionAvgPatternDf[np.logical_and(unionAvgPatternDf['time'] > 0, unionAvgPatternDf['time'] < 5e-3)]['e ' + str(int(electrode)) + ' hpf']
            sig = sig - noise_mean[i]
            if np.isnan(sig).all():
                toa[i] = np.nan
            else:
                toa[i] = unionAvgPatternDf['time'][sig.abs().argmax()]
        else:
            indices = unionAvgPatternDf['time'] > cropStartingAt # np.logical_and(unionAvgPatternDf['time'] > cropStartingAt, unionAvgPatternDf['time'] < processWindow - 4e-3)
            sig = unionAvgPatternDf.loc[indices, 'e ' + str(int(electrode)) + ' hpf']
            sig = sig - noise_mean[i]
            times = unionAvgPatternDf.loc[indices, 'time']
            toa[i] = times.to_numpy()[sig.abs().to_numpy().argmax()]
        power[i] = np.power(sig.abs().max(), 2) # per sample
        snr[i] = power[i]/noisePower[i]
    mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='power', value=power)
    mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='toa', value=toa)
    mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='snr', value=snr)
    mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='noise_mean', value=noise_mean)
    mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='noisePower', value=noisePower)
    for m,mvAvg_value in enumerate(mvAvg_values):
        mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='toa_std@MA ' + str(int(mvAvg_value)), value=toa_std[:,m])
        mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='maxSpatialCorrVal@MA ' + str(int(mvAvg_value)), value=maxCorrVal[:,m])
        mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='maxSpatialCorrDTOA@MA ' + str(int(mvAvg_value)), value=maxCorrDTOA[:,m])
        mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='maxSpatialCorr_xDiff@MA ' + str(int(mvAvg_value)), value=maxCorrElectrode_xDiff[:,m])
        mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='maxSpatialCorr_yDiff@MA ' + str(int(mvAvg_value)), value=maxCorrElectrode_yDiff[:,m])
    
    spikeRefElectrodePower = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].astype(int)==spikeRefElectrodeNo]['power'].to_numpy()
    spikeRefElectrodeToa = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].astype(int)==spikeRefElectrodeNo]['toa'].to_numpy()
    
    mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='power wrt ref', value=mappingDFOfElectrodesIn_unionAvgPatternDf['power'].to_numpy()/spikeRefElectrodePower)
    mappingDFOfElectrodesIn_unionAvgPatternDf.insert(loc=mappingDFOfElectrodesIn_unionAvgPatternDf.shape[1], column='toa wrt ref', value=mappingDFOfElectrodesIn_unionAvgPatternDf['toa'].to_numpy() - spikeRefElectrodeToa)
    refDf = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode']==spikeRefElectrodeNo]
    
    plt.figure(figsize=(10,10))
    nSpikesInRef = int(union_mappingDF[union_mappingDF['electrode']==spikeRefElectrodeNo]['nSpikes'].to_numpy()[0])
    vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['snr'].min(), mappingDFOfElectrodesIn_unionAvgPatternDf['snr'].max()
    if not(mappingDFOfElectrodesIn_unionAvgPatternDf['snr'].isna().all()):
        single_SNRs = 10*np.log10(mappingDFOfElectrodesIn_unionAvgPatternDf['snr'])-10*np.log10(nSpikesInRef)
        indices = single_SNRs >= 6
        plt.scatter(vmin=10*np.log10(vmin)-10*np.log10(nSpikesInRef), vmax=10*np.log10(vmax)-10*np.log10(nSpikesInRef), x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'][indices], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'][indices], s=20, c=single_SNRs[indices])
        plt.xlabel('um')
        plt.ylabel('um')
        plt.axis('equal')
        cbar=plt.colorbar()
        cbar.ax.set_ylabel('SNR single event [db]')#,fontsize=12)
        plt.title(f'single SNR thr = 6db')
        plt.legend()
        plt.grid()
    if not(refDf['snr'].isna().all()):
        plt.scatter(vmin=10*np.log10(vmin), vmax=10*np.log10(vmax), x=refDf['x'], y=refDf['y'], s=500, c=10*np.log10(refDf['snr']), marker='*', label='soma')
    
    plt.savefig(analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_activityPlotSingleSNR.png', dpi=150)
    plt.close()
    
    plt.figure(figsize=(30,10))
    plt.subplot(1,3,1)
    nSpikesInRef = int(union_mappingDF[union_mappingDF['electrode']==spikeRefElectrodeNo]['nSpikes'].to_numpy()[0])
    vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['snr'].min(), mappingDFOfElectrodesIn_unionAvgPatternDf['snr'].max()
    if not(mappingDFOfElectrodesIn_unionAvgPatternDf['snr'].isna().all()):
        plt.scatter(vmin=10*np.log10(vmin), vmax=10*np.log10(vmax), x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=10*np.log10(mappingDFOfElectrodesIn_unionAvgPatternDf['snr']))
        plt.xlabel('um')
        plt.ylabel('um')
        plt.axis('equal')
        cbar=plt.colorbar()
        cbar.ax.set_ylabel('SNR [db]')#,fontsize=12)
        plt.title(f'Processing gain = {str(round(10*np.log10(nSpikesInRef), 1))}db')
        plt.legend()
        plt.grid()
    if not(refDf['snr'].isna().all()):
        plt.scatter(vmin=10*np.log10(vmin), vmax=10*np.log10(vmax), x=refDf['x'], y=refDf['y'], s=500, c=10*np.log10(refDf['snr']), marker='*', label='soma')

    plt.subplot(1,3,2)
    vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['power wrt ref'].quantile(5/100), mappingDFOfElectrodesIn_unionAvgPatternDf['power wrt ref'].quantile(95/100)
    if not(mappingDFOfElectrodesIn_unionAvgPatternDf['power wrt ref'].isna().all()):
        plt.scatter(vmin=10*np.log10(vmin), vmax=10*np.log10(vmax), x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=10*np.log10(mappingDFOfElectrodesIn_unionAvgPatternDf['power wrt ref']))
        plt.xlabel('um')
        plt.ylabel('um')
        plt.axis('equal')
        cbar=plt.colorbar()
        cbar.ax.set_ylabel('power wrt ref [db]')#,fontsize=12)
        plt.legend()
        plt.grid()
    if not(refDf['power wrt ref'].isna().all()):
        plt.scatter(vmin=10*np.log10(vmin), vmax=10*np.log10(vmax), x=refDf['x'], y=refDf['y'], s=500, c=10*np.log10(refDf['power wrt ref']), marker='*', label='soma')
    
    
    plt.subplot(1,3,3)
    vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(5/100), mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(85/100)
    if not(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].isna().all()):
        plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref']/1e-6)
        plt.xlabel('um')
        plt.ylabel('um')
        plt.axis('equal')
        cbar=plt.colorbar()
        cbar.ax.set_ylabel('toa [us]')#,fontsize=12)
        plt.legend()
        plt.grid()
    if not(refDf['toa wrt ref'].isna().all()):
        plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=refDf['x'], y=refDf['y'], s=500, c=refDf['toa wrt ref']/1e-6, marker='*', label='soma')
    
    plt.savefig(analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_activityPlot.png', dpi=150)
    plt.close()

    toa_std_all_mvAvg, toa_all_mvAvg = np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0], len(mvAvg_values))), np.zeros((mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0], len(mvAvg_values)))
    branchs = mappingDFOfElectrodesIn_unionAvgPatternDf['branch'].unique().tolist()
    for m,mvAvg_value in enumerate(mvAvg_values):
        toa_std_all_mvAvg[:,m] = mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))]
        toa_all_mvAvg[:,m] = mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref']
        
        vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].quantile(0 / 100), mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].quantile(95 / 100)
        if not np.isnan(vmin):
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            
            if not(mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].isna().all()):
                plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))]/1e-6)
            if not(refDf['toa_std@MA ' + str(int(mvAvg_value))].isna().all()):
                plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=refDf['x'], y=refDf['y'], s=500, c=refDf['toa_std@MA ' + str(int(mvAvg_value))]/1e-6, marker='*', label='soma')
            plt.xlabel('um')
            plt.ylabel('um')
            plt.axis('equal')
            cbar=plt.colorbar()
            cbar.ax.set_ylabel('toa_std [us]')#,fontsize=12)
            plt.legend()
            plt.grid()
    
            plt.subplot(1,2,2)
            vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(5/100), mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(95/100)
            if not(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].isna().all()):
                plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref']/1e-6)
            if not(refDf['toa wrt ref'].isna().all()):
                plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=refDf['x'], y=refDf['y'], s=500, c=refDf['toa wrt ref']/1e-6, marker='*', label='soma')
            plt.xlabel('um')
            plt.ylabel('um')
            plt.axis('equal')
            cbar=plt.colorbar()
            cbar.ax.set_ylabel('toa [us]')#,fontsize=12)
            plt.legend()
            plt.grid()
            plt.savefig(analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_activityPlotTOA@MA_' + str(int(mvAvg_value)) + '.png', dpi=150)
            plt.close()
        
        
        if not np.isnan(vmin):
            plt.figure(figsize=(30,10))
            snr_MA = mappingDFOfElectrodesIn_unionAvgPatternDf['snr']#/nSpikesInRef*int(mvAvg_value)
            plt.subplot(1,3,1)
            plt.suptitle('window size = ' + mvAvg_value)
            vmin, vmax = snr_MA.quantile(5/100), snr_MA.quantile(95/100)
            if not(mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].isna().all()):
                eucleadianDists = np.sqrt(np.power(mappingDFOfElectrodesIn_unionAvgPatternDf['x'] - refDf['x'].to_numpy(), 2) + np.power(mappingDFOfElectrodesIn_unionAvgPatternDf['y'] - refDf['y'].to_numpy(), 2))
                corrRes = eucleadianDists.corr(mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))])
                #plt.title(f'corr = {str(round(corrRes, 2))}')
                if not(snr_MA.isna().all()):
                    plt.scatter(c=10*np.log10(snr_MA), vmin=10*np.log10(vmin), vmax=10*np.log10(vmax), x=eucleadianDists, s=20, y=mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))]/1e-6)
                else:
                    plt.scatter(x=eucleadianDists, s=20, y=mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))]/1e-6)
                plt.xlabel('euclidean distance [um]')
                plt.ylabel('toa_std [us]')
                plt.ylim([mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].quantile(0/100)/1e-6, mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].quantile(95/100)/1e-6])
                #plt.legend()
                plt.grid()
            
            plt.subplot(1,3,2)
            
            if not(mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].isna().all()):
                eucleadianDists = np.sqrt(np.power(mappingDFOfElectrodesIn_unionAvgPatternDf['x'] - refDf['x'].to_numpy(), 2) + np.power(mappingDFOfElectrodesIn_unionAvgPatternDf['y'] - refDf['y'].to_numpy(), 2))
                corrRes = eucleadianDists.corr(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'])
                #plt.title(f'corr = {str(round(corrRes, 2))}')
                if not(snr_MA.isna().all()):
                    plt.scatter(c=10*np.log10(snr_MA), vmin=10*np.log10(vmin), vmax=10*np.log10(vmax), x=eucleadianDists, s=20, y=mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref']/1e-6)
                else:
                    plt.scatter(x=eucleadianDists, s=20, y=mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref']/1e-6)
                plt.xlabel('euclidean distance [um]')
                plt.ylabel('toa [us]')
                plt.ylim([mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(0/100)/1e-6, mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(95/100)/1e-6])
                #plt.legend()
                plt.grid()
            
            plt.subplot(1,3,3)
            
            if not(mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].isna().all()):                
                corrRes = mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].corr(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'])
                #plt.title(f'corr = {str(round(corrRes, 2))}')
                if not(snr_MA.isna().all()):
                    plt.scatter(c=10*np.log10(snr_MA), vmin=10*np.log10(vmin), vmax=10*np.log10(vmax), y=mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))]/1e-6, s=20, x=mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref']/1e-6)
                    cbar=plt.colorbar()
                    cbar.ax.set_ylabel('snr [db]')#,fontsize=12)
                else:
                    plt.scatter(y=mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))]/1e-6, s=20, x=mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref']/1e-6)                
                plt.xlabel('toa [us]')
                plt.ylabel('toa_std [us]')
                plt.xlim([mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(0/100)/1e-6, mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(95/100)/1e-6])
                plt.ylim([mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].quantile(0/100)/1e-6, mappingDFOfElectrodesIn_unionAvgPatternDf['toa_std@MA ' + str(int(mvAvg_value))].quantile(95/100)/1e-6])
                #plt.legend()
                plt.grid()
                
            
            plt.savefig(analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_activityScatterPlotTOA_vs_Dist@MA_' + str(int(mvAvg_value)) + '.png', dpi=150)
            plt.close()
            
####################################################################################################################################################################################################            
####################################################################################################################################################################################################
####################################################################################################################################################################################################
            
            path2spikeFile = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_spikes_h5_DF.csv'
            spikes_h5_DF = pd.read_csv(path2spikeFile)
            spikeRefChannelNo = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'] == spikeRefElectrodeNo]['channel'].to_numpy()[0].astype(int)
            spikes_h5_DF_channel = spikes_h5_DF[spikes_h5_DF['channel']==spikeRefChannelNo]
            spikes_h5_DF_channel = spikes_h5_DF_channel.sort_values(by='time')
            cellSpikesAmps = spikes_h5_DF_channel['amplitude']
            
            
            binsWidth = 250e-6 # sec
            binsGap = 50e-6
            binsStart = np.arange(0, mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(95/100), binsWidth)
            binsStop = binsStart + binsWidth
            thr_maxTimes = 225e-6
            
    
            if m == 0:
                list_bins_polyFitTimes = list()
                bestSnrElectrodeList = list()
                for branch in branchs:
                    bestSnrElectrodeInBranchList = list()
                    for b in range(len(binsStart)):
                        indices_branch = mappingDFOfElectrodesIn_unionAvgPatternDf['branch'] == branch
                        indices_timeBin = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] >= binsStart[b]+binsGap, mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] <= binsStop[b]-binsGap)
                        indices = np.logical_and(indices_branch, indices_timeBin)
                        if False:
                            if nameOf_h5 == 'Ron.raw.h5':
                                #ref_y = refDf['y'].to_numpy()[0]
                                #print(f'refDf[y].to_numpy() = {ref_y}')
                                #print(f'indices.shape = {indices.shape}')
                                locationIndices = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['y'] >= refDf['y'].to_numpy()[0], mappingDFOfElectrodesIn_unionAvgPatternDf['x'] <= refDf['x'].to_numpy()[0])
                                
                                locationIndices_x = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['x'] >= 2600, mappingDFOfElectrodesIn_unionAvgPatternDf['x'] <= 3000)
                                locationIndices_y = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['y'] >= 900, mappingDFOfElectrodesIn_unionAvgPatternDf['y'] <= 1500)
                                squareIndices = np.logical_not(np.logical_and(locationIndices_x,locationIndices_y))
                                locationIndices = np.logical_and(locationIndices, squareIndices)
                                
                                locationIndices_x = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['x'] >= 2200, mappingDFOfElectrodesIn_unionAvgPatternDf['x'] <= 2700)
                                locationIndices_y = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['y'] >= 480, mappingDFOfElectrodesIn_unionAvgPatternDf['y'] <= 700)
                                squareIndices = np.logical_not(np.logical_and(locationIndices_x,locationIndices_y))
                                locationIndices = np.logical_and(locationIndices, squareIndices)
                                
                                locationIndices_x = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['x'] >= 2500, mappingDFOfElectrodesIn_unionAvgPatternDf['x'] <= 2600)
                                locationIndices_y = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['y'] >= 1000, mappingDFOfElectrodesIn_unionAvgPatternDf['y'] <= 1200)
                                squareIndices = np.logical_not(np.logical_and(locationIndices_x,locationIndices_y))
                                locationIndices = np.logical_and(locationIndices, squareIndices)
                                #print(f'locationIndices.shape = {locationIndices.shape}')
                                indices = np.logical_and(indices, locationIndices)
                                print('CUSTOM BEST SNR ELECTRODES SELECTION')
                            elif nameOf_h5 == 'Trace_20230706_07_58_30_spont.raw.h5':
                                locationIndices = np.logical_not(mappingDFOfElectrodesIn_unionAvgPatternDf['x'] >= 2500)
                                locationIndices = np.logical_and(locationIndices, np.logical_not(mappingDFOfElectrodesIn_unionAvgPatternDf['x'] <= 1600))
                                locationIndices = np.logical_and(locationIndices, mappingDFOfElectrodesIn_unionAvgPatternDf['y'] >= refDf['y'].to_numpy()[0])
                                indices = np.logical_and(indices, locationIndices)
                                print('CUSTOM BEST SNR ELECTRODES SELECTION')
                        
                        if indices.any():
                            mappingDF_bin = mappingDFOfElectrodesIn_unionAvgPatternDf.loc[indices]
                            electrode = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['electrode']
                            bestSnrElectrodeInBranchList.append(electrode)
                    if False:    
                        if nameOf_h5 == 'Ron.raw.h5':
                            bestSnrElectrodeInBranchList = bestSnrElectrodeInBranchList[:4] + bestSnrElectrodeInBranchList[5:8]
                            print('CUSTOM BEST SNR ELECTRODES SELECTION')
                        elif nameOf_h5 == 'Trace_20230706_07_58_30_spont.raw.h5':
                            bestSnrElectrodeInBranchList = bestSnrElectrodeInBranchList[:4] + bestSnrElectrodeInBranchList[5:7] + [bestSnrElectrodeInBranchList[8]] + [bestSnrElectrodeInBranchList[10]] + bestSnrElectrodeInBranchList[13:15]
                            print('CUSTOM BEST SNR ELECTRODES SELECTION')
                
                    bestSnrElectrodeList.append(bestSnrElectrodeInBranchList)
                
                for branch, bestSnrElectrodeInBranchList in zip(branchs, bestSnrElectrodeList):
                    plt.figure(figsize=(10,10))
                    nSpikesInRef = int(union_mappingDF[union_mappingDF['electrode']==spikeRefElectrodeNo]['nSpikes'].to_numpy()[0])
                    vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['snr'].min(), mappingDFOfElectrodesIn_unionAvgPatternDf['snr'].max()
                    if not(mappingDFOfElectrodesIn_unionAvgPatternDf['snr'].isna().all()):
                        mappingDFOfElectrodesIn_unionAvgPatternDf_bestSnr = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].isin(bestSnrElectrodeInBranchList)]
                        plt.scatter(vmin=10*np.log10(vmin), vmax=10*np.log10(vmax), x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=10*np.log10(mappingDFOfElectrodesIn_unionAvgPatternDf['snr']))
                        plt.scatter(x=mappingDFOfElectrodesIn_unionAvgPatternDf_bestSnr['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf_bestSnr['y'], s=500, marker='+', color='red', label='chosen electrodes')
                        for b,e in enumerate(bestSnrElectrodeInBranchList):
                            x, y = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'] == e]['x'], mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'] == e]['y']
                            dtoa = (mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'] == e]['toa wrt ref']).to_numpy()[0]
                            plt.text(x=x, y=y, s=str(int(e)) + ': ' + str(round(dtoa/1e-6)))#, fontsize='small')
                        plt.xlabel('um')
                        plt.ylabel('um')
                        plt.axis('equal')
                        cbar=plt.colorbar()
                        cbar.ax.set_ylabel('SNR [db]')#,fontsize=12)
                        plt.title(f'Branch = {str(int(branch))}; Processing gain = {str(round(10*np.log10(nSpikesInRef), 1))}db')
                        plt.legend()
                        plt.grid()
                
                
                    saveStr = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_branch_' + str(int(branch)) + '_bestSnrElectrodes' + '.png'
                    plt.savefig(saveStr, dpi=150)
                    print('saved ' + saveStr)
                    plt.close()
            
            if m == 0:
                path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(spikeRefElectrodeNo)) + '_mySpikesWithFeatures_DF.csv'
                #print('loading ' + path2my_mySpikeDf_withFeatures)
                _mySpikeDfRefElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                corrsOptimalRolledPolyFitTimes = _mySpikeDfRefElectrode[f'toa@MA {mvAvg_value} @e {int(spikeRefElectrodeNo)}']
                spikeRefElectrode_toa = corrsOptimalRolledPolyFitTimes.median()
                for branch, bestSnrElectrodeInBranchList in zip(branchs, bestSnrElectrodeList):
                    branchList = list()
                    for b in range(len(bestSnrElectrodeInBranchList)):
                        electrode = bestSnrElectrodeInBranchList[b]
    
                        path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(electrode)) + '_mySpikesWithFeatures_DF.csv'
                        print('loading ' + path2my_mySpikeDf_withFeatures)
                        _mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                        
                        m_list = list()
                        for mb,mvAvg_value_b in enumerate(mvAvg_values):
                            corrsOptimalRolledPolyFitTimes = _mySpikeDfElectrode[f'toa@MA {mvAvg_value_b} @e {int(electrode)}']
                            corrsOptimalRolledPolyFitTimes = corrsOptimalRolledPolyFitTimes - spikeRefElectrode_toa
                            corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfElectrode['time'] - _mySpikeDfElectrode['time'][0]
                            m_list.append([corrsOptimalRolledPolyFitTimes, corrsOptimalRolledPolyFitTimes_tVec])
                        branchList.append(m_list)
                    list_bins_polyFitTimes.append(branchList)
            
            for branchIdx, branch in enumerate(branchs):
                bestSnrElectrodeInBranchList = bestSnrElectrodeList[branchIdx]
                plt.figure(figsize=(20,20))
                path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(spikeRefElectrodeNo)) + '_mySpikesWithFeatures_DF.csv'
                #print('loading ' + path2my_mySpikeDf_withFeatures)
                _mySpikeDfRefElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                corrsOptimalRolledPolyFitTimes = _mySpikeDfRefElectrode[f'toa@MA {mvAvg_value} @e {int(spikeRefElectrodeNo)}']
                spikeRefElectrode_toa = corrsOptimalRolledPolyFitTimes.median()
                print(f'spikeRefElectrode_toa = {spikeRefElectrode_toa}')
                print(f'corrsOptimalRolledPolyFitTimes[0] = {corrsOptimalRolledPolyFitTimes[0]}')
                corrsOptimalRolledPolyFitTimes = corrsOptimalRolledPolyFitTimes - spikeRefElectrode_toa
                corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfRefElectrode['time'] - _mySpikeDfRefElectrode['time'][0]
                
                #plt.figure()
                #plt.plot(corrsOptimalRolledPolyFitTimes_tVec, corrsOptimalRolledPolyFitTimes/1e-6)
                #plt.title(f'spikeRefElectrode_toa = {spikeRefElectrode_toa/1e-6} us')
                #plt.xlabel('sec')
                #plt.ylabel('usec')
                #saveStr = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_polyFit@MA_' + str(int(mvAvg_value)) + '.png'
                #plt.savefig(saveStr, dpi=150)
                #plt.close()
                
                
                maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                median_maxTimes = np.median(maxTimes)
                maxTimes[np.abs(maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                
                #plt.subplot(len(binsStart), 1, b+1)
                plt.plot(corrsOptimalRolledPolyFitTimes_tVec, maxTimes/1e-6, label=f'toa = {str(round(0/1e-6))}')
                
                for b in range(len(bestSnrElectrodeInBranchList)):
                    corrsOptimalRolledPolyFitTimes, corrsOptimalRolledPolyFitTimes_tVec = list_bins_polyFitTimes[branchIdx][b][m]
                    toa_wrt_ref = corrsOptimalRolledPolyFitTimes.median()
                    
                    maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                    median_maxTimes = np.median(maxTimes)
                    maxTimes[np.abs(maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                    
                    #plt.subplot(len(binsStart), 1, b+1)
                    
                    
                    if b == 0:
                        _bestSnrToas_df = pd.DataFrame(columns=['time'], data=corrsOptimalRolledPolyFitTimes_tVec)
                    
                    columnName = f'toa = {str(round(toa_wrt_ref/1e-6))} us'
                    if not(columnName in _bestSnrToas_df.columns):
                        _bestSnrToas_df.insert(loc=_bestSnrToas_df.shape[1], column=columnName, value=maxTimes)
                    
                        plt.plot(corrsOptimalRolledPolyFitTimes_tVec, maxTimes/1e-6, label=f'toa = {str(round(toa_wrt_ref/1e-6))}')
                    
                plt.xlabel('sec')
                plt.ylabel('us')
                plt.grid()
                plt.legend(loc='upper right')
                plt.title(f'Branch = {str(int(branch))}; window = {str(int(mvAvg_value))}; ' + f'detections within {str(round(thr_maxTimes/1e-6))} us')
                saveStr = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo)  + '_branch_' + str(int(branch)) +  '_bestSnrToas@MA_' + str(int(mvAvg_value)) + '.png'
                _bestSnrToas_df.to_csv(saveStr[:saveStr.find('.png')] + '.csv')
                plt.savefig(saveStr, dpi=150)
                plt.close()
                print('saved ' + saveStr)
                
                plt.figure()
                for column in _bestSnrToas_df.columns:
                    if column == 'time':
                        continue
                    data = (_bestSnrToas_df[column] - _bestSnrToas_df[column].median()).to_numpy()    
                    plt.hist(data/1e-6,bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=False,label=column)
                plt.grid()
                plt.legend(loc='upper right')
                plt.title(f'Branch = {str(int(branch))}; window = {str(int(mvAvg_value))}; ' + f'detections within {str(round(thr_maxTimes/1e-6))} us')
                plt.xlabel('us')
                saveStr = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo)  + '_branch_' + str(int(branch)) +  '_bestSnrToasCDF@MA_' + str(int(mvAvg_value)) + '.png'
                plt.savefig(saveStr, dpi=150)
                plt.close()
                print('saved ' + saveStr)
                
                
                plt.figure(figsize=(6.4*2,4.8*len(binsStart)))
                #plt.figure()#figsize=(20*10*corrsOptimalRolledPolyFitTimes_tVec.to_numpy()[-1]/12000,20))
                previous_toa_wrt_ref = 0.0
                #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(spikeRefElectrodeNo)) + '_mySpikesWithFeatures_DF.csv'
                #print('loading ' + path2my_mySpikeDf_withFeatures)
                #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                corrsOptimalRolledPolyFitTimes = _mySpikeDfRefElectrode[f'toa@MA {mvAvg_value} @e {int(spikeRefElectrodeNo)}'] - spikeRefElectrode_toa
                corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfRefElectrode['time'] - _mySpikeDfRefElectrode['time'][0]
                previous_maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                median_maxTimes = np.median(previous_maxTimes)
                previous_maxTimes[np.abs(previous_maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                for b in range(len(bestSnrElectrodeInBranchList)):
                    #indices = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] >= binsStart[b]+binsGap, mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] <= binsStop[b]-binsGap)
                    #mappingDF_bin = mappingDFOfElectrodesIn_unionAvgPatternDf.loc[indices]
                    #electrode = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['electrode']
                    electrode = bestSnrElectrodeInBranchList[b]
                    #toa_wrt_ref = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa wrt ref']
                    #toa = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa']
                    
                    
                    
                    #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(electrode)) + '_mySpikesWithFeatures_DF.csv'
                    #print('loading ' + path2my_mySpikeDf_withFeatures)
                    #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                    #corrsOptimalRolledPolyFitTimes = _mySpikeDfElectrode[f'toa@MA {mvAvg_value} @e {int(electrode)}']
                    #corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfElectrode['time'] - _mySpikeDfElectrode['time'][0]
                    
                    corrsOptimalRolledPolyFitTimes, corrsOptimalRolledPolyFitTimes_tVec = list_bins_polyFitTimes[branchIdx][b][m]
                    toa_wrt_ref = corrsOptimalRolledPolyFitTimes.median()
                    mean_toa_wrt_ref_diff = toa_wrt_ref - previous_toa_wrt_ref
                    
                    maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                    median_maxTimes = np.median(maxTimes)
                    maxTimes[np.abs(maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                    
                    dynamic_toa_wrt_ref_diff = maxTimes - previous_maxTimes
                    dynamic_toa_wrt_ref_wrt_mean = dynamic_toa_wrt_ref_diff/mean_toa_wrt_ref_diff
                    
                    #plt.subplot(len(binsStart), 1, b+1)
                    #plt.plot(corrsOptimalRolledPolyFitTimes_tVec, dynamic_toa_wrt_ref_wrt_mean, linewidth=1.0, label=f'toa = {str(round(toa_wrt_ref/1e-6))}')
                    if b > 0:
                        plt.subplot(len(binsStart)-1, 2, 2*b-1)
                        corr = pd.Series(previous_dynamic_toa_wrt_ref_wrt_mean).corr(pd.Series(dynamic_toa_wrt_ref_wrt_mean))
                        plt.scatter(x=previous_dynamic_toa_wrt_ref_wrt_mean, y=dynamic_toa_wrt_ref_wrt_mean,s=1, label=f'toa = {str(round(toa_wrt_ref/1e-6))}; corr = {str(round(corr, 2))}')
                        plt.grid()
                        plt.legend(loc='upper right')
                        plt.ylabel('segment n velocity change')
                        plt.xlabel('segment n-1 velocity change')
                        
                        plt.subplot(len(binsStart)-1, 2, 2*b)
                        corr = pd.Series(previous_dynamic_toa_wrt_ref_diff).corr(pd.Series(dynamic_toa_wrt_ref_diff))
                        plt.scatter(x=previous_dynamic_toa_wrt_ref_diff/1e-6, y=dynamic_toa_wrt_ref_diff/1e-6,s=1, label=f'toa = {str(round(toa_wrt_ref/1e-6))}; corr = {str(round(corr, 2))}')
                        plt.grid()
                        plt.legend(loc='upper right')
                        plt.ylabel('segment n [us]')
                        plt.xlabel('segment n-1 [us]')
                    if b == 1:
                        plt.title(f'Branch = {str(int(branch))}; Segments; window = {str(int(mvAvg_value))}; ' + f'detections within {str(round(thr_maxTimes/1e-6))} us')
                    
                    previous_toa_wrt_ref = toa_wrt_ref
                    previous_maxTimes = maxTimes
                    previous_dynamic_toa_wrt_ref_wrt_mean = dynamic_toa_wrt_ref_wrt_mean
                    previous_dynamic_toa_wrt_ref_diff = dynamic_toa_wrt_ref_diff
                    
                    
                plt.tight_layout()
                
                saveStr = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo)  + '_branch_' + str(int(branch)) +  '_bestSnrToasSegments@MA_' + str(int(mvAvg_value)) + '.png'
                plt.savefig(saveStr, dpi=150)
                plt.close()
                print('saved ' + saveStr)
                
                plt.figure(figsize=(6.4*3,4.8*len(binsStart)))
                #plt.figure()#figsize=(20*10*corrsOptimalRolledPolyFitTimes_tVec.to_numpy()[-1]/12000,20))
                previous_toa_wrt_ref = 0.0
                #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(spikeRefElectrodeNo)) + '_mySpikesWithFeatures_DF.csv'
                #print('loading ' + path2my_mySpikeDf_withFeatures)
                #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                corrsOptimalRolledPolyFitTimes = _mySpikeDfRefElectrode[f'toa@MA {mvAvg_value} @e {int(spikeRefElectrodeNo)}'] - spikeRefElectrode_toa
                corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfRefElectrode['time'] - _mySpikeDfRefElectrode['time'][0]
                previous_maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                median_maxTimes = np.median(previous_maxTimes)
                previous_maxTimes[np.abs(previous_maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                for b in range(len(bestSnrElectrodeInBranchList)):
                    #indices = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] >= binsStart[b]+binsGap, mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] <= binsStop[b]-binsGap)
                    #mappingDF_bin = mappingDFOfElectrodesIn_unionAvgPatternDf.loc[indices]
                    #electrode = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['electrode']
                    electrode = bestSnrElectrodeInBranchList[b]
                    #toa_wrt_ref = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa wrt ref']
                    #toa = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa']
                    
                    
                    
                    #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(electrode)) + '_mySpikesWithFeatures_DF.csv'
                    #print('loading ' + path2my_mySpikeDf_withFeatures)
                    #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                    #corrsOptimalRolledPolyFitTimes = _mySpikeDfElectrode[f'toa@MA {mvAvg_value} @e {int(electrode)}']
                    #corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfElectrode['time'] - _mySpikeDfElectrode['time'][0]
                    
                    corrsOptimalRolledPolyFitTimes, corrsOptimalRolledPolyFitTimes_tVec = list_bins_polyFitTimes[branchIdx][b][m]
                    toa_wrt_ref = corrsOptimalRolledPolyFitTimes.median()
                    mean_toa_wrt_ref_diff = toa_wrt_ref - previous_toa_wrt_ref
                    
                    maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                    median_maxTimes = np.median(maxTimes)
                    maxTimes[np.abs(maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                    
                    dynamic_toa_wrt_ref_diff = maxTimes - previous_maxTimes
                    dynamic_toa_wrt_ref_wrt_mean = dynamic_toa_wrt_ref_diff/mean_toa_wrt_ref_diff
                    #dynamic_toa_wrt_ref_wrt_mean = (dynamic_toa_wrt_ref_wrt_mean-np.nanmean(dynamic_toa_wrt_ref_wrt_mean))/np.nanstd(dynamic_toa_wrt_ref_wrt_mean)
                    
                    #plt.subplot(len(binsStart), 1, b+1)
                    #plt.plot(corrsOptimalRolledPolyFitTimes_tVec, dynamic_toa_wrt_ref_wrt_mean, linewidth=1.0, label=f'toa = {str(round(toa_wrt_ref/1e-6))}')
                    if b > 0:
                        plt.subplot(len(binsStart)-1, 1, b)
                        corr = pd.Series(previous_dynamic_toa_wrt_ref_wrt_mean).corr(pd.Series(dynamic_toa_wrt_ref_wrt_mean))
                        plt.plot(corrsOptimalRolledPolyFitTimes_tVec, dynamic_toa_wrt_ref_wrt_mean, label=f'toa = {str(round(toa_wrt_ref/1e-6))}; corr = {str(round(corr, 2))}; std = {str(round(np.nanstd(dynamic_toa_wrt_ref_wrt_mean),2))}')
                        plt.plot(corrsOptimalRolledPolyFitTimes_tVec, previous_dynamic_toa_wrt_ref_wrt_mean, label=f'toa = {str(round(previous_toa_wrt_ref/1e-6))}; std = {str(round(np.nanstd(previous_dynamic_toa_wrt_ref_wrt_mean),2))}')
                        plt.grid()
                        plt.legend(loc='upper right')
                        plt.ylabel('normalized velocity change')
                        plt.xlabel('sec')
                        #ylim = np.nanquantile(np.abs(dynamic_toa_wrt_ref_wrt_mean), 0.95)
                        #plt.ylim([-ylim, ylim])
                        
                    if b == 1:
                        plt.title(f'Branch = {str(int(branch))}; Segments; window = {str(int(mvAvg_value))}; ' + f'detections within {str(round(thr_maxTimes/1e-6))} us')
                    
                    previous_toa_wrt_ref = toa_wrt_ref
                    previous_maxTimes = maxTimes
                    previous_dynamic_toa_wrt_ref_wrt_mean = dynamic_toa_wrt_ref_wrt_mean
                    previous_dynamic_toa_wrt_ref_diff = dynamic_toa_wrt_ref_diff
                    
                    
                plt.tight_layout()
                
                saveStr = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo)  + '_branch_' + str(int(branch)) +  '_bestSnrToasSegmentsTS@MA_' + str(int(mvAvg_value)) + '.png'
                plt.savefig(saveStr, dpi=150)
                plt.close()
                print('saved ' + saveStr)
                
                
                #################################
                plt.figure(figsize=(6.4*3,4.8*len(binsStart)))
                #plt.figure()#figsize=(20*10*corrsOptimalRolledPolyFitTimes_tVec.to_numpy()[-1]/12000,20))
                previous_toa_wrt_ref = 0.0
                #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(spikeRefElectrodeNo)) + '_mySpikesWithFeatures_DF.csv'
                #print('loading ' + path2my_mySpikeDf_withFeatures)
                #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                corrsOptimalRolledPolyFitTimes = _mySpikeDfRefElectrode[f'toa@MA {mvAvg_value} @e {int(spikeRefElectrodeNo)}'] - spikeRefElectrode_toa
                corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfRefElectrode['time'] - _mySpikeDfRefElectrode['time'][0]
                previous_maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                median_maxTimes = np.median(previous_maxTimes)
                previous_maxTimes[np.abs(previous_maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                for b in range(len(bestSnrElectrodeInBranchList)):
                    #indices = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] >= binsStart[b]+binsGap, mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] <= binsStop[b]-binsGap)
                    #mappingDF_bin = mappingDFOfElectrodesIn_unionAvgPatternDf.loc[indices]
                    #electrode = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['electrode']
                    electrode = bestSnrElectrodeInBranchList[b]
                    #toa_wrt_ref = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa wrt ref']
                    #toa = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa']
                    
                    
                    
                    #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(electrode)) + '_mySpikesWithFeatures_DF.csv'
                    #print('loading ' + path2my_mySpikeDf_withFeatures)
                    #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                    #corrsOptimalRolledPolyFitTimes = _mySpikeDfElectrode[f'toa@MA {mvAvg_value} @e {int(electrode)}']
                    #corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfElectrode['time'] - _mySpikeDfElectrode['time'][0]
                    
                    corrsOptimalRolledPolyFitTimes, corrsOptimalRolledPolyFitTimes_tVec = list_bins_polyFitTimes[branchIdx][b][m]
                    toa_wrt_ref = corrsOptimalRolledPolyFitTimes.median()
                    
                    
                    maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                    median_maxTimes = np.nanmedian(maxTimes)
                    maxTimes = maxTimes - median_maxTimes
                    maxTimes[np.abs(maxTimes) > thr_maxTimes] = 0.0
                    maxTimes[np.isnan(maxTimes)] = 0
                    
                    maxTimesAC = np.correlate(maxTimes, maxTimes, mode='full')
                    maxTimesAC = maxTimesAC / np.correlate(np.ones_like(maxTimes), np.ones_like(maxTimes), mode='full')
                    eventVec = np.arange(len(maxTimesAC)) - len(maxTimes) + 1
                    
                    acl = int(len(maxTimesAC)*0.25)
                    maxTimesAC = maxTimesAC[:-acl]
                    maxTimesAC = maxTimesAC[acl:]
                    eventVec = eventVec[:-acl]
                    eventVec = eventVec[acl:]
                    
                    maxTimesAC = maxTimesAC/maxTimesAC.max()
                    plt.subplot(len(binsStart), 1, b+1)
                    plt.plot(eventVec, 10*np.log10(np.abs(maxTimesAC)), label=f'toa = {str(round(toa_wrt_ref/1e-6))}')
                    
                    plt.grid()
                    plt.legend(loc='upper right')
                    plt.ylabel('normalized autocorr [db]')
                    plt.xlabel('events')
                        
                        
                    if b == 0:
                        plt.title(f'Branch = {str(int(branch))}; AutoCorr; window = {str(int(mvAvg_value))}; ' + f'detections within {str(round(thr_maxTimes/1e-6))} us')
       
                plt.tight_layout()
                
                saveStr = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo)  + '_branch_' + str(int(branch)) +  '_bestSnrToasAutoCorr@MA_' + str(int(mvAvg_value)) + '.png'
                plt.savefig(saveStr, dpi=150)
                plt.close()
                print('saved ' + saveStr)
                #################################
                
                
                #plt.figure(figsize=(6.4,4.8*len(binsStart)))
                plt.figure(figsize=(20,20))
                previous_toa_wrt_ref = 0.0
                #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(spikeRefElectrodeNo)) + '_mySpikesWithFeatures_DF.csv'
                #print('loading ' + path2my_mySpikeDf_withFeatures)
                #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                corrsOptimalRolledPolyFitTimes = _mySpikeDfRefElectrode[f'toa@MA {mvAvg_value} @e {int(spikeRefElectrodeNo)}'] - spikeRefElectrode_toa
                corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfRefElectrode['time'] - _mySpikeDfRefElectrode['time'][0]
                previous_maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                median_maxTimes = np.median(previous_maxTimes)
                previous_maxTimes[np.abs(previous_maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                for b in range(len(bestSnrElectrodeInBranchList)):
                    #indices = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] >= binsStart[b]+binsGap, mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] <= binsStop[b]-binsGap)
                    #mappingDF_bin = mappingDFOfElectrodesIn_unionAvgPatternDf.loc[indices]
                    #electrode = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['electrode']
                    electrode = bestSnrElectrodeInBranchList[b]
                    #toa_wrt_ref = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa wrt ref']
                    #toa = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa']
                    
                    
                    
                    #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(electrode)) + '_mySpikesWithFeatures_DF.csv'
                    #print('loading ' + path2my_mySpikeDf_withFeatures)
                    #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                    #corrsOptimalRolledPolyFitTimes = _mySpikeDfElectrode[f'toa@MA {mvAvg_value} @e {int(electrode)}']
                    #corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfElectrode['time'] - _mySpikeDfElectrode['time'][0]
                    corrsOptimalRolledPolyFitTimes, corrsOptimalRolledPolyFitTimes_tVec = list_bins_polyFitTimes[branchIdx][b][m]
                    toa_wrt_ref = corrsOptimalRolledPolyFitTimes.median()
                    mean_toa_wrt_ref_diff = toa_wrt_ref# - previous_toa_wrt_ref
                    
                    maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                    median_maxTimes = np.median(maxTimes)
                    maxTimes[np.abs(maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                    
                    dynamic_toa_wrt_ref_diff = maxTimes - previous_maxTimes
                    #dynamic_toa_wrt_ref_wrt_mean = dynamic_toa_wrt_ref_diff/mean_toa_wrt_ref_diff
                    
                    #plt.subplot(len(binsStart), 1, b+1)
                    plt.plot(corrsOptimalRolledPolyFitTimes_tVec, dynamic_toa_wrt_ref_diff/1e-6, linewidth=0.3, label=f'toa = {str(round(toa_wrt_ref/1e-6))}')
                    
                    #previous_toa_wrt_ref = toa_wrt_ref
                    #previous_maxTimes = maxTimes
                plt.ylabel('toa change')    
                plt.grid()
                plt.legend(loc='upper right')
                plt.ylabel('us')
                    #if b == len(binsStart)-1:
                plt.xlabel('sec')
                    
                    #if b == 0:
                plt.title(f'Branch = {str(int(branch))}; From cell; window = {str(int(mvAvg_value))}; ' + f'detections within {str(round(thr_maxTimes/1e-6))} us')
                
                
                plt.tight_layout()
                
                saveStr = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo)  + '_branch_' + str(int(branch)) +  '_bestSnrToasFromCell@MA_' + str(int(mvAvg_value)) + '.png'
                plt.savefig(saveStr, dpi=150)
                plt.close()
                print('saved ' + saveStr)
                
                
                plt.figure(figsize=(20,20))
                previous_toa_wrt_ref = 0.0
                #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(spikeRefElectrodeNo)) + '_mySpikesWithFeatures_DF.csv'
                #print('loading ' + path2my_mySpikeDf_withFeatures)
                #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                corrsOptimalRolledPolyFitTimes = _mySpikeDfRefElectrode[f'toa@MA {mvAvg_value} @e {int(spikeRefElectrodeNo)}'] - spikeRefElectrode_toa
                corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfRefElectrode['time'] - _mySpikeDfRefElectrode['time'][0]
                previous_maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                median_maxTimes = np.median(previous_maxTimes)
                previous_maxTimes[np.abs(previous_maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                for b in range(len(bestSnrElectrodeInBranchList)):
                    #indices = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] >= binsStart[b]+binsGap, mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] <= binsStop[b]-binsGap)
                    #mappingDF_bin = mappingDFOfElectrodesIn_unionAvgPatternDf.loc[indices]
                    #electrode = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['electrode']
                    electrode = bestSnrElectrodeInBranchList[b]
                    #toa_wrt_ref = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa wrt ref']
                    #toa = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa']
                    
                    
                    
                    #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(electrode)) + '_mySpikesWithFeatures_DF.csv'
                    #print('loading ' + path2my_mySpikeDf_withFeatures)
                    #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                    #corrsOptimalRolledPolyFitTimes = _mySpikeDfElectrode[f'toa@MA {mvAvg_value} @e {int(electrode)}']
                    #corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfElectrode['time'] - _mySpikeDfElectrode['time'][0]
                    corrsOptimalRolledPolyFitTimes, corrsOptimalRolledPolyFitTimes_tVec = list_bins_polyFitTimes[branchIdx][b][m]
                    toa_wrt_ref = corrsOptimalRolledPolyFitTimes.median()
                    mean_toa_wrt_ref_diff = toa_wrt_ref# - previous_toa_wrt_ref
                    
                    maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                    median_maxTimes = np.median(maxTimes)
                    maxTimes[np.abs(maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                    
                    dynamic_toa_wrt_ref_diff = maxTimes - previous_maxTimes
                    dynamic_toa_wrt_ref_wrt_mean = dynamic_toa_wrt_ref_diff/mean_toa_wrt_ref_diff
                    
                    #plt.subplot(len(binsStart), 1, b+1)
                    plt.scatter(x=cellSpikesAmps, y=dynamic_toa_wrt_ref_wrt_mean, s=1, label=f'toa = {str(round(toa_wrt_ref/1e-6))}')
                    
                    #previous_toa_wrt_ref = toa_wrt_ref
                    #previous_maxTimes = maxTimes
                    
                plt.xlabel('amps')
                plt.xlim([cellSpikesAmps.quantile(5/100), cellSpikesAmps.quantile(95/100)])
                plt.ylabel('toa change')
                plt.grid()
                plt.legend(loc='upper right')
                plt.title(f'Branch = {str(int(branch))}; From cell; window = {str(int(mvAvg_value))}; ' + f'detections within {str(round(thr_maxTimes/1e-6))} us')
                saveStr = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo)  + '_branch_' + str(int(branch)) + '_bestSnrToasFromCellScatter@MA_' + str(int(mvAvg_value)) + '.png'
                plt.savefig(saveStr, dpi=150)
                plt.close()
                print('saved ' + saveStr)
                
                plt.figure(figsize=(20,20))
                previous_toa_wrt_ref = 0.0
                #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(spikeRefElectrodeNo)) + '_mySpikesWithFeatures_DF.csv'
                #print('loading ' + path2my_mySpikeDf_withFeatures)
                #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                corrsOptimalRolledPolyFitTimes = _mySpikeDfRefElectrode[f'toa@MA {mvAvg_value} @e {int(spikeRefElectrodeNo)}'] - spikeRefElectrode_toa
                corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfRefElectrode['time'] - _mySpikeDfRefElectrode['time'][0]
                previous_maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                median_maxTimes = np.median(previous_maxTimes)
                previous_maxTimes[np.abs(previous_maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                for b in range(len(bestSnrElectrodeInBranchList)):
                    #indices = np.logical_and(mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] >= binsStart[b]+binsGap, mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'] <= binsStop[b]-binsGap)
                    #mappingDF_bin = mappingDFOfElectrodesIn_unionAvgPatternDf.loc[indices]
                    #electrode = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['electrode']
                    electrode = bestSnrElectrodeInBranchList[b]
                    #toa_wrt_ref = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa wrt ref']
                    #toa = mappingDF_bin.loc[mappingDF_bin['snr'].idxmax()]['toa']
                    
                    
                    
                    #path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(int(electrode)) + '_mySpikesWithFeatures_DF.csv'
                    #print('loading ' + path2my_mySpikeDf_withFeatures)
                    #_mySpikeDfElectrode = pd.read_csv(path2my_mySpikeDf_withFeatures)
                    #corrsOptimalRolledPolyFitTimes = _mySpikeDfElectrode[f'toa@MA {mvAvg_value} @e {int(electrode)}']
                    #corrsOptimalRolledPolyFitTimes_tVec = _mySpikeDfElectrode['time'] - _mySpikeDfElectrode['time'][0]
                    corrsOptimalRolledPolyFitTimes, corrsOptimalRolledPolyFitTimes_tVec = list_bins_polyFitTimes[branchIdx][b][m]
                    toa_wrt_ref = corrsOptimalRolledPolyFitTimes.median()
                    mean_toa_wrt_ref_diff = toa_wrt_ref# - previous_toa_wrt_ref
                    
                    maxTimes = corrsOptimalRolledPolyFitTimes.to_numpy()
                    median_maxTimes = np.median(maxTimes)
                    maxTimes[np.abs(maxTimes-median_maxTimes) > thr_maxTimes] = np.nan
                    
                    dynamic_toa_wrt_ref_diff = maxTimes - previous_maxTimes
                    dynamic_toa_wrt_ref_wrt_mean = dynamic_toa_wrt_ref_diff/mean_toa_wrt_ref_diff
                    
                    #plt.subplot(len(binsStart), 1, b+1)
                    corrsOptimalRolledPolyFitTimes_tVec_diff = corrsOptimalRolledPolyFitTimes_tVec.diff()
                    plt.scatter(x=corrsOptimalRolledPolyFitTimes_tVec_diff, y=dynamic_toa_wrt_ref_wrt_mean, s=1, label=f'toa = {str(round(toa_wrt_ref/1e-6))}')
                    
                    #previous_toa_wrt_ref = toa_wrt_ref
                    #previous_maxTimes = maxTimes
                    
                plt.xlabel('diff [sec]')
                plt.xlim([corrsOptimalRolledPolyFitTimes_tVec_diff.quantile(5/100), corrsOptimalRolledPolyFitTimes_tVec_diff.quantile(95/100)])
                plt.ylabel('toa change')
                plt.grid()
                plt.legend(loc='upper right')
                plt.title(f'Branch = {str(int(branch))}; From cell; window = {str(int(mvAvg_value))}; ' + f'detections within {str(round(thr_maxTimes/1e-6))} us')
                saveStr = analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo)  + '_branch_' + str(int(branch)) +  '_bestSnrToasFromCellScatterTimes@MA_' + str(int(mvAvg_value)) + '.png'
                plt.savefig(saveStr, dpi=150)
                plt.close()
                print('saved ' + saveStr)
                    
            
####################################################################################################################################################################################################            
####################################################################################################################################################################################################
####################################################################################################################################################################################################
            
            
        vmin, vmax = 10*np.log10(mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorrVal@MA ' + str(int(mvAvg_value))].abs().quantile(0 / 100)), 10*np.log10(mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorrVal@MA ' + str(int(mvAvg_value))].abs().quantile(100 / 100))
        if not np.isnan(vmin):
            plt.figure(figsize=(20,20))
            plt.subplot(2,2,1)
            
            if not(mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorrVal@MA ' + str(int(mvAvg_value))].isna().all()):
                plt.scatter(vmin=vmin, vmax=vmax , x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=10*np.log10(mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorrVal@MA ' + str(int(mvAvg_value))].abs()))
            if not(refDf['maxSpatialCorrVal@MA ' + str(int(mvAvg_value))].isna().all()):
                plt.scatter(vmin=vmin, vmax=vmax, x=refDf['x'], y=refDf['y'], s=500, c=10*np.log10(refDf['maxSpatialCorrVal@MA ' + str(int(mvAvg_value))].abs()), marker='*', label='soma')
            plt.xlabel('um')
            plt.ylabel('um')
            plt.axis('equal')
            cbar=plt.colorbar()
            cbar.ax.set_ylabel('max corr [db]')#,fontsize=12)
            plt.legend()
            plt.grid()
    
            plt.subplot(2,2,2)
            vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorrDTOA@MA ' + str(int(mvAvg_value))].quantile(5/100), mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorrDTOA@MA ' + str(int(mvAvg_value))].quantile(95/100)
            if not(mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorrDTOA@MA ' + str(int(mvAvg_value))].isna().all()):
                plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorrDTOA@MA ' + str(int(mvAvg_value))]/1e-6)
            if not(refDf['maxSpatialCorrDTOA@MA ' + str(int(mvAvg_value))].isna().all()):
                plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=refDf['x'], y=refDf['y'], s=500, c=refDf['maxSpatialCorrDTOA@MA ' + str(int(mvAvg_value))]/1e-6, marker='*', label='soma')
            plt.xlabel('um')
            plt.ylabel('um')
            plt.axis('equal')
            cbar=plt.colorbar()
            cbar.ax.set_ylabel('max corr dtoa [us]')#,fontsize=12)
            plt.legend()
            plt.grid()
            
            plt.subplot(2,2,3)
            vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorr_xDiff@MA ' + str(int(mvAvg_value))].quantile(0 / 100), mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorr_xDiff@MA ' + str(int(mvAvg_value))].quantile(100 / 100)
            if not(mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorr_xDiff@MA ' + str(int(mvAvg_value))].isna().all()):
                plt.scatter(vmin=vmin, vmax=vmax , x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorr_xDiff@MA ' + str(int(mvAvg_value))])
            if not(refDf['maxSpatialCorr_xDiff@MA ' + str(int(mvAvg_value))].isna().all()):
                plt.scatter(vmin=vmin, vmax=vmax, x=refDf['x'], y=refDf['y'], s=500, c=refDf['maxSpatialCorr_xDiff@MA ' + str(int(mvAvg_value))], marker='*', label='soma')
            plt.xlabel('um')
            plt.ylabel('um')
            plt.axis('equal')
            cbar=plt.colorbar()
            cbar.ax.set_ylabel('max corr xDiff')#,fontsize=12)
            plt.legend()
            plt.grid()
    
            plt.subplot(2,2,4)
            vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorr_yDiff@MA ' + str(int(mvAvg_value))].quantile(0 / 100), mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorr_yDiff@MA ' + str(int(mvAvg_value))].quantile(100 / 100)
            if not(mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorr_yDiff@MA ' + str(int(mvAvg_value))].isna().all()):
                plt.scatter(vmin=vmin, vmax=vmax , x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=mappingDFOfElectrodesIn_unionAvgPatternDf['maxSpatialCorr_yDiff@MA ' + str(int(mvAvg_value))])
            if not(refDf['maxSpatialCorr_yDiff@MA ' + str(int(mvAvg_value))].isna().all()):
                plt.scatter(vmin=vmin, vmax=vmax, x=refDf['x'], y=refDf['y'], s=500, c=refDf['maxSpatialCorr_yDiff@MA ' + str(int(mvAvg_value))], marker='*', label='soma')
            plt.xlabel('um')
            plt.ylabel('um')
            plt.axis('equal')
            cbar=plt.colorbar()
            cbar.ax.set_ylabel('max corr yDiff')#,fontsize=12)
            plt.legend()
            plt.grid()
            
            plt.savefig(analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_activityPlotMaxSpatialCorr@MA_' + str(int(mvAvg_value)) + '.png', dpi=150)
            plt.close()
    
    plt.figure(figsize=(20,10))
    mvAvg_values_array = np.asarray([int(m) for m in mvAvg_values])
    mvAvg_values_sortIndices = mvAvg_values_array.argsort()
    mvAvg_values_array = mvAvg_values_array[mvAvg_values_sortIndices]
    toa_std_all_mvAvg = toa_std_all_mvAvg[:, mvAvg_values_sortIndices]
    print(f'mvAvg_values_array: {mvAvg_values_array}')
    plt.subplot(1,2,1)
    for e in range(toa_std_all_mvAvg.shape[0]):
        plt.plot(mvAvg_values_array, toa_std_all_mvAvg[e]/1e-6, linewidth=0.2)        
    plt.xlabel('window size')
    plt.ylabel('toa_std [us]')
    plt.grid()
    plt.subplot(1,2,2)
    for e in range(toa_std_all_mvAvg.shape[0]):
        plt.plot(mvAvg_values_array, np.log(np.abs(toa_std_all_mvAvg[e]/toa_all_mvAvg[e])), linewidth=0.2)
    plt.xlabel('window size')
    plt.ylabel('log(toa_std over toa)')
    plt.grid()
    
    plt.savefig(analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_toa_std_vs_MA_' + '.png', dpi=150)
    plt.close()
    
    plt.figure(figsize=(10,10))
    vmin, vmax = mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(5/100), mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref'].quantile(85/100)
    plt.xlabel('um')
    plt.ylabel('um')
    plt.axis('equal')
    
    plt.grid()
        
    
    for i,e,x,y in zip(np.arange(mappingDFOfElectrodesIn_unionAvgPatternDf.shape[0]), mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'], mappingDFOfElectrodesIn_unionAvgPatternDf['x'], mappingDFOfElectrodesIn_unionAvgPatternDf['y']):
        plt.text(x=x, y=y, s=str(int(e)), fontsize='xx-small')
    plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=mappingDFOfElectrodesIn_unionAvgPatternDf['x'], y=mappingDFOfElectrodesIn_unionAvgPatternDf['y'], s=20, c=mappingDFOfElectrodesIn_unionAvgPatternDf['toa wrt ref']/1e-6)
    plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=refDf['x'], y=refDf['y'], s=500, c=refDf['toa wrt ref']/1e-6, marker='*', label='soma')
    plt.xlabel('um')
    plt.ylabel('um')
    cbar=plt.colorbar()
    cbar.ax.set_ylabel('toa [us]')#,fontsize=12)
    plt.legend()
    #plt.xlim([1200, 2800])
    #plt.ylim([0, 2500])
    
    plt.grid()
    plt.savefig(analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_activityPlotNames.png', dpi=150)
    plt.close()
    
    mappingDFOfElectrodesInGoodSNR_unionAvgPatternDf = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].isin(ListOfGoodSNR_electrodesForPlot)]
    
    plt.figure(figsize=(10,10))
    vmin, vmax = mappingDFOfElectrodesInGoodSNR_unionAvgPatternDf['toa wrt ref'].quantile(5/100), mappingDFOfElectrodesInGoodSNR_unionAvgPatternDf['toa wrt ref'].quantile(85/100)
    plt.grid()
    
    
    for i,e,x,y in zip(np.arange(mappingDFOfElectrodesInGoodSNR_unionAvgPatternDf.shape[0]), mappingDFOfElectrodesInGoodSNR_unionAvgPatternDf['electrode'], mappingDFOfElectrodesInGoodSNR_unionAvgPatternDf['x'], mappingDFOfElectrodesInGoodSNR_unionAvgPatternDf['y']):
        plt.text(x=x, y=y, s=str(int(e)), fontsize='xx-small')
    plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=mappingDFOfElectrodesInGoodSNR_unionAvgPatternDf['x'], y=mappingDFOfElectrodesInGoodSNR_unionAvgPatternDf['y'], s=20, c=mappingDFOfElectrodesInGoodSNR_unionAvgPatternDf['toa wrt ref']/1e-6)
    plt.scatter(vmin=vmin/1e-6, vmax=vmax/1e-6, x=refDf['x'], y=refDf['y'], s=500, c=refDf['toa wrt ref']/1e-6, marker='*', label='soma')
    plt.axis('equal')
    cbar=plt.colorbar()
    cbar.ax.set_ylabel('toa [us]')#,fontsize=12)
    plt.legend()
    plt.xlabel('um')
    plt.ylabel('um')
    #plt.xlim([1200, 2800])
    #plt.ylim([0, 2500])
    plt.title('electrodes with single spike identification')
    plt.grid()
    plt.show()
    
    
    #######   Electrode investigation ##############
    
    # 21022
    if enableElectrodeInvestigation:
        investigationElectrode = [spikeRefElectrodeNo] + investigationElectrode
        for ei,processedElectrodeNo in enumerate(investigationElectrode):
            mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'] == processedElectrodeNo]
            cstr = 'e ' + str(processedElectrodeNo) + ' hpf'
            Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
            path2my_mySpikeDf_withSamples = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithSamples_DF.csv'
            if not(os.path.isfile(path2my_mySpikeDf_withSamples)):
                timeSeriesDf = createTimeSeriesDf(filename, analysisLibrary, nameOf_h5, configuration, union_mappingDF, spikeRefElectrodeNo, processedElectrodeNo, tVec, fs)
            else:
                timeSeriesDf = None
            print(f'{datetime.datetime.now()}: starting processElectrode with spikeRef = {spikeRefElectrodeNo} and processed = {processedElectrodeNo}; electrode {ei} out of {len(electrodeProcessList)}')
            processElectrode(None, filename, analysisLibrary, nameOf_h5, configuration, union_mappingDF, spikeRefElectrodeNo, processedElectrodeNo, timeShiftVsRef, timeSeriesDf, spikes_h5_DF, fs, True, False, Id_str, use_hpfAvg, process_hpf, processWindow)
            
    ################################################
    mappingDFOfElectrodesIn_unionAvgPatternDf.sort_values(by=['electrode'], inplace=True)
    if enablePlotPerElectrode:
        for i,electrode in enumerate(mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].tolist()):
            fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*1*1), sharex=True, sharey=True)
            x,y = union_mappingDF[union_mappingDF['electrode'] == electrode]['x'].to_numpy()[0], union_mappingDF[union_mappingDF['electrode'] == electrode]['y'].to_numpy()[0]
            indices = unionAvgPatternDf['time'] > cropStartingAt
            sig = unionAvgPatternDf.loc[indices, 'e ' + str(int(electrode)) + ' hpf']
            noise_mean = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].astype(int)==electrode]['noise_mean'].to_numpy()[0]
            noisePower = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].astype(int)==electrode]['noisePower'].to_numpy()[0]
            power = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].astype(int)==electrode]['power'].to_numpy()[0]
            toa = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].astype(int)==electrode]['toa'].to_numpy()[0]
            snr = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].astype(int)==electrode]['snr'].to_numpy()[0]
            sig = sig - noise_mean
            maxTime = unionAvgPatternDf.loc[indices, 'time'].to_numpy()[-1]
            plt.plot(unionAvgPatternDf.loc[indices, 'time']/1e-3, sig, linewidth=0.5, label='avg(e ' + str(int(electrode)) + ' hpf); ' + f'SNR={str(round(10*np.log10(snr),2))} db')
            plt.hlines(y=[-np.sqrt(noisePower), np.sqrt(noisePower)], xmin=[0,0], xmax=[maxTime/1e-3, maxTime/1e-3], color='black', linestyle='dashed')
            plt.hlines(y=[-np.sqrt(power), np.sqrt(power)], xmin=[(toa-1e-3)/1e-3,(toa-1e-3)/1e-3], xmax=[(toa+1e-3)/1e-3,(toa+1e-3)/1e-3], color='red', linestyle='dashed')
            plt.xlabel('ms')
            plt.title(f'e '+ str(int(electrode)) + f'(x,y)=({x},{y})')
            plt.grid()
            plt.legend()
            plt.savefig(analysisLibrary + nameOf_h5 + '_allConf_' + 'wrt_e_' + str(spikeRefElectrodeNo) + '_electrode_' + str(electrode) + '_ts.png', dpi=150)
            plt.close()
        
        fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*1*1), sharex=True, sharey=True)
        for i,electrode in enumerate(mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].tolist()):
            
            x,y = union_mappingDF[union_mappingDF['electrode'] == electrode]['x'].to_numpy()[0], union_mappingDF[union_mappingDF['electrode'] == electrode]['y'].to_numpy()[0]
            indices = unionAvgPatternDf['time'] > cropStartingAt
            sig = unionAvgPatternDf.loc[indices, 'e ' + str(int(electrode)) + ' hpf']
            noise_mean = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].astype(int)==electrode]['noise_mean'].to_numpy()[0]
            sig = sig - noise_mean
            plt.plot(unionAvgPatternDf.loc[indices, 'time']/1e-3, sig, linewidth=0.5)#, label='avg(e ' + str(int(electrode)) + ' hpf)')
            plt.xlabel('ms')
            #plt.title(f'e '+ str(int(electrode)) + f'(x,y)=({x},{y})')
            plt.grid()
            #plt.legend()
        plt.title('all electrodes avg')
        plt.show()
        
        fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*1*1), sharex=True, sharey=True)
        for i,electrode in enumerate(mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].tolist()):
            
            x,y = union_mappingDF[union_mappingDF['electrode'] == electrode]['x'].to_numpy()[0], union_mappingDF[union_mappingDF['electrode'] == electrode]['y'].to_numpy()[0]
            indices = unionAvgPatternDf['time'] > cropStartingAt
            sig = unionAvgPatternDf.loc[indices, 'e ' + str(int(electrode)) + ' hpf']
            noise_mean = mappingDFOfElectrodesIn_unionAvgPatternDf[mappingDFOfElectrodesIn_unionAvgPatternDf['electrode'].astype(int)==electrode]['noise_mean'].to_numpy()[0]
            sig = sig - noise_mean
            ts = 20*np.log10(sig) - 20*np.log10(sig).median()
            plt.plot(unionAvgPatternDf.loc[indices, 'time']/1e-3, ts, linewidth=0.2)#, label='avg(e ' + str(int(electrode)) + ' hpf)')
            plt.xlabel('ms')
            plt.ylabel('db')
            #plt.title(f'e '+ str(int(electrode)) + f'(x,y)=({x},{y})')
            plt.grid()
            #plt.legend()
        plt.ylim([-10,10])
        plt.title('all electrodes power')
        plt.show()
    
    
