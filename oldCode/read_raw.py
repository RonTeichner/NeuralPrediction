import json
from PIL import Image
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
import platform
from read_raw_func import *
 
plt.close('all')
axx, ab = 16/3, 9/3
plt.close('all')
if platform.system() == 'Linux':
    path2Dropbox = '/media/ront/4d9d1135-ebb8-4cdb-b6b8-81eb1dd36cb1/media/ront/MXBIO/'#'/media/ront/4d9d1135-ebb8-4cdb-b6b8-81eb1dd36cb1/media/ront/DropboxTechnionBackup/'
else:
    path2Dropbox = '/Users/ron.teichner/Data/MXBIO/'


# read file
#print(f'inputs from keyboard are {sys.argv}')
if len(sys.argv) > 1:
    jsonFileName = sys.argv[1]
else:
    jsonFileName = './runNeuron18_18878.json'
print(f'json file name is {jsonFileName}')

with open(jsonFileName, 'r') as myfile:
    runCommands=json.loads(myfile.read())['config'][0]

path2ExpWithinDropbox = runCommands["path2ExpWithinDropbox"]
nameOf_h5 = runCommands["nameOf_h5"]
path2metrics_data_xlsxWithinDropbox = runCommands["path2metrics_data_xlsxWithinDropbox"]
path2exp = path2Dropbox + path2ExpWithinDropbox
filename = path2exp + nameOf_h5
path2metrics_data_xlsx = path2Dropbox + path2metrics_data_xlsxWithinDropbox
if 'analysisName' in runCommands.keys():
    analysisName = runCommands["analysisName"]
else:
    analysisName = ''
analysisLibrary = path2exp + nameOf_h5 + '_' + analysisName + '_ronAnalysis/'

saveImages = runCommands['saveImages'] == 1
trajectoriesAndExit = runCommands['trajectoriesAndExit'] == 1
processElectrode_enableFigures = runCommands['processElectrode_enableFigures'] == 1
save_mySpikes = runCommands['save_mySpikes'] == 1
use_hpfAvg = runCommands["use_hpfAvg"]
process_hpf = runCommands["process_hpf"]
processWindow = runCommands["processWindow_readRaw"]
nConfigurations = runCommands["nConfigurations"]
enableGet2Know_hFile = False
enableSummaryPlots = False
processType = runCommands["processType_readRaw"]

configurations = [str(i).zfill(4) for i in range(nConfigurations)]
nSpikesThr = runCommands["nSpikesThr"] # at least 25% of spikes in Neuron were identified by the system
#channel number = 789
Neurons = [runCommands["Neuron"]]
branchs = -1#[[1,2,6,11,19]]#1
if 'stimuliElectrodeNo' in runCommands.keys():
    stimuliElectrodeNo = runCommands["stimuliElectrodeNo"]
else:
    if stimuliElectrodeNo == -1:
        stimuliElectrodeNo = None

preDefinedTimeShiftVsRefLists = 0e-3#[[[0e-3,1.0e-3,1.7e-3],[0e-3,1.0e-3,1.15e-3,1.6e-3],[0e-3,1.2e-3],[0e-3,1.1e-3],[0e-3,0.7e-3]]]#[0, 1.2e-3, 1.5e-3, 2.5e-3]
channelProcessList = -1
'''
processType = 'processNeuronsAndBranches' # {'processStimuli', 'processNeuronsAndBranches'}
path2exp = path2Dropbox + 'MXWBio/18628_Jan24_2023/230208/18628_Jan24_2023/AxonTracking/000002/'
filename = path2exp + 'data.raw.h5'
path2metrics_data_xlsx = path2Dropbox + 'MXWBio/18628_Jan24_2023/metrics_data.xlsx'
configuration = '0021'
nSpikesThr = 0.0 # at least 25% of spikes in Neuron were identified by the system
#channel number = 789
Neurons = [14]
branchs = -1#[[1,2,6,11,19]]#1
stimuliElectrodeNo = None
preDefinedTimeShiftVsRefLists = 1e-3#[[[0e-3,1.0e-3,1.7e-3],[0e-3,1.0e-3,1.15e-3,1.6e-3],[0e-3,1.2e-3],[0e-3,1.1e-3],[0e-3,0.7e-3]]]#[0, 1.2e-3, 1.5e-3, 2.5e-3]


processType = 'processNeuronsAndBranches' # {'processStimuli', 'processNeuronsAndBranches'}
path2exp = path2Dropbox + 'MXWBio/18628_Jan24_2023/230208/18628_Jan24_2023/AxonTracking/000002/'
filename = path2exp + 'data.raw.h5'
path2metrics_data_xlsx = path2Dropbox + 'MXWBio/18628_Jan24_2023/metrics_data.xlsx'
configuration = '0004'#'0021'
nSpikesThr = 0.0 # at least 25% of spikes in Neuron were identified by the system
Neurons = [14]
branchs = [[1,2,6,11,19]]#1
stimuliElectrodeNo = None
preDefinedTimeShiftVsRefLists = [[[0e-3,1.0e-3,1.7e-3],[0e-3,1.0e-3,1.15e-3,1.6e-3],[0e-3,1.2e-3],[0e-3,1.1e-3],[0e-3,0.7e-3]]]#[0, 1.2e-3, 1.5e-3, 2.5e-3]

path2exp = path2Dropbox + 'MXWBio/recordingsWithStimuli/'
filename = path2exp + 'Trace_20230518_09_58_14(STIMe3412trigger_IPI150ms_50mV).raw.h5'
path2metrics_data_xlsx = None
configuration = '0000'
channelProcessList = [88]
preDefinedTimeShiftVsRefLists = [[[-0.1e-3,1.55e-3]]]

processType = 'processStimuli' # {'processStimuli', 'processNeuronsAndBranches'}
path2exp = path2Dropbox + 'MXWBio/recordingsWithStimuli/'
filename = path2exp + 'Trace_20230518_11_39_01(STIMe5697axon_IPI150ms_200mV).raw.h5'
path2metrics_data_xlsx = None
configuration = '0000'
nSpikesThr = 0.25 # at least 25% of spikes in Neuron were identified by the system
channelProcessList = [59, 76, 56, 54, 34, 424]
stimuliElectrodeNo = 5697
preDefinedTimeShiftVsRefLists = [[[0.0e-3, 3.0e-3, 3.0e-3, 3.0e-3, 3.0e-3, 3.0e-3, 3.0e-3]]]

processType = 'processNeuronsAndBranches' # {'processStimuli', 'processNeuronsAndBranches'}
path2exp = path2Dropbox + 'MXWBio/recordingsWithStimuli/rec_18812_Apr30_2023/'
filename = path2exp + 'Trace_20230523_13_56_16_Spontaneous.raw.h5'
path2metrics_data_xlsx = path2exp + 'metrics_data.xlsx'
configuration = '0000'
Neurons = [18]
branchs = -1 # all branches
nSpikesThr = 0.25 # at least 25% of spikes in Neuron were identified by the system
preDefinedTimeShiftVsRefLists = None

processType = 'processNeuronsAndBranches' # {'processStimuli', 'processNeuronsAndBranches'}
plotActivity = True #(plotting the activity of all channels) 
path2exp = path2Dropbox + 'MXWBio/recordingsWithStimuli/rec_18812_Apr30_2023/'
filename = path2exp + 'Trace_20230523_13_56_16_Spontaneous.raw.h5'
path2metrics_data_xlsx = path2exp + 'metrics_data.xlsx'
configuration = '0000'
Neurons = [18]
channelProcessList = -1
branchs = -1 # all branches
nSpikesThr = 0.0 # at least 25% of spikes in Neuron were identified by the system
preDefinedTimeShiftVsRefLists = None
'''
'''
if 'STIMe' in filename:
    startE = 5+filename.find('STIMe')
    if 'trigger' in filename:
        stopE = filename.find('trigger')
    else:
        stopE = filename.find('axon')
    stimuliElectrodeNo = int(filename[startE:stopE])
else:
    stimuliElectrodeNo = None
'''    
    
with h5py.File(filename,'r') as hf:
    hf.visit(print)

for configuration in configurations:
    if enableGet2Know_hFile:
        rawPointers = ['/data_store/data' + configuration + '/groups/routed/raw']#, '/recordings/data' + configuration + '/well000/groups/routed/raw', '/wells/well000/data' + configuration + '/groups/routed/raw']
        with h5py.File(filename, mode='r') as h5f:
            for rawPointer in rawPointers:
                rawData = np.array(h5f[rawPointer][:])
                print(rawPointer + f' is of shape {rawData.shape}')
                rawRecordings = rawData.astype(float)
        
        print('')
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
                rawData = np.array(h5f[rawPointer][:])
                print(rawPointer + f' is of shape {rawData.shape}; Trigger channels are {rawData}')
                triggerChannels = rawData
        
        print('')
        mapping = '/data_store/data' + configuration + '/settings/mapping'
        with h5py.File(filename, mode='r') as h5f:
            rawPointer = mapping
            rawData = np.array(h5f[rawPointer][:])
            mappingDF = pd.DataFrame(columns=['channel', 'electrode', 'x', 'y', 'isTrigger'])
            for i in range(rawData.shape[0]):
                mappingDF.loc[len(mappingDF.index)] = np.asarray([rawData[i][0], rawData[i][1], rawData[i][2], rawData[i][3], rawData[i][0] in triggerChannels])
            print(rawPointer + f' is of shape {rawData.shape} representing the mapping of {rawData.shape} electrodes')
            print(rawPointer + f'[0] is of len {len(rawData[0])} representing the mapping of channel {rawData[0][0]}, electrode {rawData[0][1]} which is located at (x,y) = {rawData[0][2]},{rawData[0][3]}')
          
        print('')
        rawPointers = ['/data_store/data' + configuration + '/groups/routed/channels']
        with h5py.File(filename, mode='r') as h5f:
            for rawPointer in rawPointers:
                rawData = np.array(h5f[rawPointer][:])
                print(rawPointer + f' is of shape {rawData.shape}; just {rawData.shape} numbers in the range {rawData.min()}-{rawData.max()}')
        
        
        print('')
        rawPointers = ['/data_store/data' + configuration + '/groups/routed/frame_nos']
        with h5py.File(filename, mode='r') as h5f:
            for rawPointer in rawPointers:
                rawData = np.array(h5f[rawPointer][:])
                print(rawPointer + f' is of shape {rawData.shape} - frame numbers from which data was recorded')
                tVec = rawData/fs # sec
                
        print('')
        rawPointers = ['/data_store/data' + configuration + '/groups/routed/triggered']
        with h5py.File(filename, mode='r') as h5f:
            for rawPointer in rawPointers:
                rawData = np.array(h5f[rawPointer][:])
                if rawData[0]:
                    print('The recorded data was only stored around spikes')
                else:
                    print('All the recorded data was stored')
                    
        print('')
        rawPointers = ['/data_store/data' + configuration + '/groups/routed/trigger_channels']
        with h5py.File(filename, mode='r') as h5f:
            for rawPointer in rawPointers:
                rawData = np.array(h5f[rawPointer][:])
                print(rawPointer + f' is of shape {rawData.shape}; Trigger channels are {rawData}')
                
        print('')
        rawPointers = ['/data_store/data' + configuration + '/groups/routed/trigger_post']
        with h5py.File(filename, mode='r') as h5f:
            for rawPointer in rawPointers:
                rawData = np.array(h5f[rawPointer][:])[0]
                print(rawPointer + f' is of shape {rawData.shape} and is {rawData}')
                nPostTrigger = rawData
                
        print('')
        rawPointers = ['/data_store/data' + configuration + '/groups/routed/trigger_pre']
        with h5py.File(filename, mode='r') as h5f:
            for rawPointer in rawPointers:
                rawData = np.array(h5f[rawPointer][:])[0]
                print(rawPointer + f' is of shape {rawData.shape} and is {rawData}')
                nPreTrigger = rawData
        
        print('')
        rawPointers = ['/data_store/data' + configuration + '/spikes']
        if os.path.isfile(path2spikeFile):
            spikes_h5_DF = pd.read_csv(path2spikeFile)
        else:
            with h5py.File(filename, mode='r') as h5f:
                for rawPointer in rawPointers:
                    rawData = np.array(h5f[rawPointer][:])
                    print(rawPointer + f' is of shape {rawData.shape} and represents all spikes metadata')
                    spikes_h5_DF = pd.DataFrame(columns=['time', 'channel', 'amplitude'])
                    for i in range(rawData.shape[0]):
                        spikes_h5_DF.loc[len(spikes_h5_DF.index)] = np.asarray([rawData[i][0]/fs, rawData[i][1], rawData[i][2]])
                spikes_h5_DF.to_csv(path2spikeFile)    
    else:    
        ###############################################################################################################
        enableReadCompleteRaw = False
        if enableReadCompleteRaw:
            rawPointers = ['/data_store/data' + configuration + '/groups/routed/raw']#, '/recordings/data' + configuration + '/well000/groups/routed/raw', '/wells/well000/data' + configuration + '/groups/routed/raw']
            with h5py.File(filename, mode='r') as h5f:
                for rawPointer in rawPointers:
                    rawData = np.array(h5f[rawPointer][:])
                    print(rawPointer + f' is of shape {rawData.shape}')
                    rawRecordings = rawData.astype(float)
                
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
                    if torch.cuda.is_available():
                        spikes_h5_DF = pd.DataFrame(data=np.zeros((rawData.shape[0], 3)), columns=['time', 'channel', 'amplitude'])
                        for i in range(rawData.shape[0]):
                            print(f'spikes_h5_DF: {i} out of {rawData.shape[0]}')
                            spikes_h5_DF.loc[len(spikes_h5_DF.index)] = np.asarray([rawData[i][0]/fs, rawData[i][1], rawData[i][2]])
                    else:
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
            mappingDF.to_csv(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_mapping.csv')
        
        #if processType == 'processStimuli' and not(stimuliElectrodeNo is None):
        #    stimuliElectrodeChannelNo = mappingDF[mappingDF['electrode'] == stimuliElectrodeNo]['channel'].to_numpy()[0].astype(int)
        
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
        
        if processType == 'processStimuli':
            Neurons = [stimuliElectrodeNo]
        
        if not(processType == 'processStimuli') and branchs == -1:
            branchs = list()
            
        for ni, Neuron in enumerate(Neurons):
            
            if processType == 'processStimuli':
                NeuronElectrodeNo = Neuron
                branchs = [[-1]]
                if channelProcessList == -1:# process all channels
                    NeuronChannelNo = mappingDF[mappingDF['electrode'] == NeuronElectrodeNo]['channel'].to_numpy()[0].astype(int)
                    channelProcessList = mappingDF[np.logical_not(mappingDF['channel'] == NeuronChannelNo)]['channel'].tolist()
            elif processType == 'processNeuronAndAllElectrodes':
                branchs = [[-1]]
                NeuronElectrodeNo = metrics_data_Neuron[metrics_data_Neuron['neuron']==Neuron]['Electrode Number'].to_numpy()[0]  
                electrodeProcessList = mappingDF[np.logical_not(mappingDF['electrode'] == NeuronElectrodeNo)]['electrode'].tolist()
                electrodeProcessList = list(set(electrodeProcessList))
                electrodeProcessList = [int(r) for r in electrodeProcessList]
            elif processType == 'processNeuronsAndBranches':
                NeuronElectrodeNo = metrics_data_Neuron[metrics_data_Neuron['neuron']==Neuron]['Electrode Number'].to_numpy()[0]    
                if len(branchs) == 0:# process al branches
                    branchs.append(metrics_data_Axon[metrics_data_Axon['neuron'] == Neuron]['branch'].unique().tolist())
            
            spikeRefElectrodeNo = NeuronElectrodeNo
            for bi, branch in enumerate(branchs[ni]):
                if processType == 'processStimuli':
                    electrodes_on_branch = mappingDF[mappingDF['channel'].isin(channelProcessList)]['electrode'].tolist()
                    electrodes_on_branch = [int(electrode) for electrode in electrodes_on_branch]
                elif processType == 'processNeuronAndAllElectrodes':
                    electrodes_on_branch = electrodeProcessList
                elif processType == 'processNeuronsAndBranches':
                    electrodes_on_branch = metrics_data_Axon[np.logical_and(metrics_data_Axon['neuron'] == Neuron, metrics_data_Axon['branch'] == branch)]['electrode']
                    electrodes_on_branch = electrodes_on_branch[electrodes_on_branch > 0].to_numpy().tolist()
                
                if preDefinedTimeShiftVsRefLists is None:
                    preDefinedTimeShiftVsRefList = [0]*len([NeuronElectrodeNo] + electrodes_on_branch)
                elif type(preDefinedTimeShiftVsRefLists)==float:
                    preDefinedTimeShiftVsRefList = [preDefinedTimeShiftVsRefLists]*len([NeuronElectrodeNo] + electrodes_on_branch)
                else:
                    preDefinedTimeShiftVsRefList = preDefinedTimeShiftVsRefLists[ni][bi]
                
                if trajectoriesAndExit or saveImages:
                    trajectoryOnly = True
                    dst_rollingDict, dst_rollingSpatialDict = dict(), dict()
                    for ei, branchElectrodeNo, timeShiftVsRef in zip(np.arange(1+len(electrodes_on_branch)).tolist(), [NeuronElectrodeNo] + electrodes_on_branch, preDefinedTimeShiftVsRefList):
                        processedElectrodeNo, timeShiftVsRef = branchElectrodeNo, timeShiftVsRef
                        if saveImages:
                            mvAvg_values = list(set([fn[fn.find('_movingAvg_')+11:fn.find('_corrsOptimal')] for fn in os.listdir(analysisLibrary) if '_movingAvg_' in fn and not('_spatial_' in fn)]))
                            print(f'mvAvg_values {mvAvg_values}')
                            factor = 0.5
                            
                            path2OptimalPatternIamge = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_optimizedPattern.png'
                            path2_corrsOptimalPattern = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPattern.png'
                            path2_heatmaps = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPatternHeatmap.png'
                            path2_heatmapsExtreme = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPatternHeatmap_extreame.png'
                            path2_heatmapsMax = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPatternMaxMarkers.png'
                            
                            if os.path.isfile(path2OptimalPatternIamge):
                                optimalPatternIm = Image.open(path2OptimalPatternIamge).convert("L")
                                optimalPatternIm = optimalPatternIm.resize((int(optimalPatternIm.width*factor), int(optimalPatternIm.height*factor)))
                            else:
                                optimalPatternIm = Image.new('L', (optimalPatternIm.width, optimalPatternIm.height))
                            
                                                    
                            if os.path.isfile(path2_corrsOptimalPattern):
                                corrsOptimalPatternIm = Image.open(path2_corrsOptimalPattern).convert("L")
                                corrsOptimalPatternIm = corrsOptimalPatternIm.resize((int(corrsOptimalPatternIm.width*factor), int(corrsOptimalPatternIm.height*factor)))
                            else:
                                corrsOptimalPatternIm = Image.new('L', (corrsOptimalPatternIm.width, corrsOptimalPatternIm.height))
                            
                            if os.path.isfile(path2_heatmaps):
                                heatMapsIm = Image.open(path2_heatmaps).convert("L")
                                heatMapsIm = heatMapsIm.resize((int(heatMapsIm.width*factor), int(heatMapsIm.height*factor)))
                            else:
                                heatMapsIm = Image.new('L', (heatMapsIm.width, heatMapsIm.height))
                            
                            if os.path.isfile(path2_heatmapsExtreme):
                                heatmapsExtremeIm = Image.open(path2_heatmapsExtreme).convert("L")
                                heatmapsExtremeIm = heatmapsExtremeIm.resize((int(heatmapsExtremeIm.width*factor), int(heatmapsExtremeIm.height*factor)))
                            else:
                                heatmapsExtremeIm = Image.new('L', (heatmapsExtremeIm.width, heatmapsExtremeIm.height))
                            
                            if os.path.isfile(path2_heatmapsExtreme):
                                heatmapsMaxIm = Image.open(path2_heatmapsMax).convert("L")
                                heatmapsMaxIm = heatmapsMaxIm.resize((int(heatmapsMaxIm.width*factor), int(heatmapsMaxIm.height*factor)))
                            else:
                                heatmapsMaxIm = Image.new('L', (heatmapsMaxIm.width, heatmapsMaxIm.height))
                            
                            corrsOptimalPatternIm_fit_heatmapsMaxIm = corrsOptimalPatternIm.resize((heatmapsMaxIm.width, corrsOptimalPatternIm.height))
                            
                            if ei == 0:
                                electrodeRes = 17.5
                                width0, height0 = optimalPatternIm.width, optimalPatternIm.height
                                width1, height1 = corrsOptimalPatternIm.width, corrsOptimalPatternIm.height
                                width2, height2 = heatMapsIm.width, heatMapsIm.height
                                width3, height3 = heatmapsExtremeIm.width, heatmapsExtremeIm.height
                                width4, height4 = heatmapsMaxIm.width, heatmapsMaxIm.height + corrsOptimalPatternIm_fit_heatmapsMaxIm.height
                                min_x = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['x'].min()
                                max_x = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['x'].max()
                                min_y = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['y'].min()
                                max_y = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['y'].max()
                                dstWidth0, dstHeight0 = int((max_x + electrodeRes - min_x)/electrodeRes*width0), int((max_y + electrodeRes - min_y)/electrodeRes*height0)
                                dstWidth1, dstHeight1 = int((max_x + electrodeRes - min_x)/electrodeRes*width1), int((max_y + electrodeRes - min_y)/electrodeRes*height1)
                                dstWidth2, dstHeight2 = int((max_x + electrodeRes - min_x)/electrodeRes*width2), int((max_y + electrodeRes - min_y)/electrodeRes*height2)
                                dstWidth3, dstHeight3 = int((max_x + electrodeRes - min_x)/electrodeRes*width3), int((max_y + electrodeRes - min_y)/electrodeRes*height3)
                                dstWidth4, dstHeight4 = int((max_x + electrodeRes - min_x)/electrodeRes*width4), int((max_y + electrodeRes - min_y)/electrodeRes*height4)
                                print('creating dst images')
                                dst0, dst1, dst2 = Image.new('L', (dstWidth0, dstHeight0)), Image.new('L', (dstWidth1, dstHeight1)), Image.new('L', (dstWidth2, dstHeight2))
                                dst3 = Image.new('L', (dstWidth3, dstHeight3))
                                dst4 = Image.new('L', (dstWidth4, dstHeight4))
                            x, y = mappingDF[mappingDF['electrode'] == processedElectrodeNo]['x'], mappingDF[mappingDF['electrode'] == processedElectrodeNo]['y']
                            dst0.paste(optimalPatternIm, (int((x-min_x)/electrodeRes*width0), int((y-min_y)/electrodeRes*height0)))
                            dst1.paste(corrsOptimalPatternIm, (int((x-min_x)/electrodeRes*width1), int((y-min_y)/electrodeRes*height1)))
                            dst2.paste(heatMapsIm, (int((x-min_x)/electrodeRes*width2), int((y-min_y)/electrodeRes*height2)))
                            dst3.paste(heatmapsExtremeIm, (int((x-min_x)/electrodeRes*width3), int((y-min_y)/electrodeRes*height3)))
                            dst4.paste(heatmapsMaxIm, (int((x-min_x)/electrodeRes*width4), int((y-min_y)/electrodeRes*height4)))
                            dst4.paste(corrsOptimalPatternIm_fit_heatmapsMaxIm, (int((x-min_x)/electrodeRes*width4), int((y-min_y)/electrodeRes*height4+heatmapsMaxIm.height)))
                            
                            # Rolling values images:
                            for rollingValue in mvAvg_values:
                                path2_corrsOptimalPattern = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_movingAvg_' + rollingValue + '_corrsOptimalPattern.png'
                                path2_heatmapsMax = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_movingAvg_' + rollingValue + '_corrsOptimalPatternMaxMarkers.png'
                                
                                if os.path.isfile(path2_corrsOptimalPattern):
                                    corrsOptimalPatternIm = Image.open(path2_corrsOptimalPattern).convert("L")
                                    corrsOptimalPatternIm = corrsOptimalPatternIm.resize((int(corrsOptimalPatternIm.width*factor), int(corrsOptimalPatternIm.height*factor)))
                                else:
                                    corrsOptimalPatternIm = Image.new('L', (corrsOptimalPatternIm.width, corrsOptimalPatternIm.height))
                                
                                if os.path.isfile(path2_heatmapsMax):    
                                    heatmapsMaxIm = Image.open(path2_heatmapsMax).convert("L")
                                    heatmapsMaxIm = heatmapsMaxIm.resize((int(heatmapsMaxIm.width*factor), int(heatmapsMaxIm.height*factor)))
                                else:
                                    heatmapsMaxIm = Image.new('L', (heatmapsMaxIm.width, heatmapsMaxIm.height))
                                
                                corrsOptimalPatternIm_fit_heatmapsMaxIm = corrsOptimalPatternIm.resize((heatmapsMaxIm.width, corrsOptimalPatternIm.height))
                                
                                if ei == 0:
                                    electrodeRes = 17.5
                                    width1, height1 = corrsOptimalPatternIm.width, corrsOptimalPatternIm.height
                                    width4, height4 = heatmapsMaxIm.width, heatmapsMaxIm.height + corrsOptimalPatternIm_fit_heatmapsMaxIm.height
                                    min_x = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['x'].min()
                                    max_x = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['x'].max()
                                    min_y = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['y'].min()
                                    max_y = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['y'].max()                                    
                                    dstWidth4, dstHeight4 = int((max_x + electrodeRes - min_x)/electrodeRes*width4), int((max_y + electrodeRes - min_y)/electrodeRes*height4)
                                    print('creating dst images rooling value ' + rollingValue)
                                    dst_rollingDict[rollingValue] = Image.new('L', (dstWidth4, dstHeight4))
                                
                                x, y = mappingDF[mappingDF['electrode'] == processedElectrodeNo]['x'], mappingDF[mappingDF['electrode'] == processedElectrodeNo]['y']
                                dst_rollingDict[rollingValue].paste(heatmapsMaxIm, (int((x-min_x)/electrodeRes*width4), int((y-min_y)/electrodeRes*height4)))
                                dst_rollingDict[rollingValue].paste(corrsOptimalPatternIm_fit_heatmapsMaxIm, (int((x-min_x)/electrodeRes*width4), int((y-min_y)/electrodeRes*height4+heatmapsMaxIm.height)))
                                
                            # Rolling and spatial values images:
                            for rollingValue in mvAvg_values:
                                path2_corrsOptimalPattern = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_movingAvg_' + rollingValue + '_spatial_corrsOptimalPattern.png'
                                path2_heatmapsMax = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_movingAvg_' + rollingValue + '_spatial_corrsOptimalPatternMaxMarkers.png'
                                
                                if os.path.isfile(path2_corrsOptimalPattern):    
                                    corrsOptimalPatternIm = Image.open(path2_corrsOptimalPattern).convert("L")
                                    corrsOptimalPatternIm = corrsOptimalPatternIm.resize((int(corrsOptimalPatternIm.width*factor), int(corrsOptimalPatternIm.height*factor)))
                                else:
                                    corrsOptimalPatternIm = Image.new('L', (corrsOptimalPatternIm.width, corrsOptimalPatternIm.height))
                                
                                if os.path.isfile(path2_heatmapsMax):    
                                    heatmapsMaxIm = Image.open(path2_heatmapsMax).convert("L")
                                    heatmapsMaxIm = heatmapsMaxIm.resize((int(heatmapsMaxIm.width*factor), int(heatmapsMaxIm.height*factor)))
                                else:
                                    heatmapsMaxIm = Image.new('L', (heatmapsMaxIm.width, heatmapsMaxIm.height))
                                    
                                corrsOptimalPatternIm_fit_heatmapsMaxIm = corrsOptimalPatternIm.resize((heatmapsMaxIm.width, corrsOptimalPatternIm.height))
                                
                                if ei == 0:
                                    electrodeRes = 17.5
                                    width1, height1 = corrsOptimalPatternIm.width, corrsOptimalPatternIm.height
                                    width4, height4 = heatmapsMaxIm.width, heatmapsMaxIm.height + corrsOptimalPatternIm_fit_heatmapsMaxIm.height
                                    min_x = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['x'].min()
                                    max_x = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['x'].max()
                                    min_y = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['y'].min()
                                    max_y = mappingDF[mappingDF['electrode'].isin(electrodes_on_branch + [NeuronElectrodeNo])]['y'].max()                                    
                                    dstWidth4, dstHeight4 = int((max_x + electrodeRes - min_x)/electrodeRes*width4), int((max_y + electrodeRes - min_y)/electrodeRes*height4)
                                    print('creating dst images rooling value ' + rollingValue)
                                    dst_rollingSpatialDict[rollingValue] = Image.new('L', (dstWidth4, dstHeight4))
                                
                                x, y = mappingDF[mappingDF['electrode'] == processedElectrodeNo]['x'], mappingDF[mappingDF['electrode'] == processedElectrodeNo]['y']
                                dst_rollingSpatialDict[rollingValue].paste(heatmapsMaxIm, (int((x-min_x)/electrodeRes*width4), int((y-min_y)/electrodeRes*height4)))
                                dst_rollingSpatialDict[rollingValue].paste(corrsOptimalPatternIm_fit_heatmapsMaxIm, (int((x-min_x)/electrodeRes*width4), int((y-min_y)/electrodeRes*height4+heatmapsMaxIm.height)))
                            
                            

                            print(f'added electrode {processedElectrodeNo} which is {ei} out of {1+len(electrodes_on_branch)} to image')    
                            
                        else:
                            if processType == 'processStimuli':
                                Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
                            elif processType == 'processNeuronAndAllElectrodes':
                                Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
                            elif processType == 'processNeuronsAndBranches':
                                Id_str = f'(c,n,b,e)={(int(configuration), Neuron, branch, processedElectrodeNo)}'
                            nSpikesInProcessedElectrode = mappingDF[mappingDF['electrode']==processedElectrodeNo]['nSpikes'].to_numpy()[0]
                            if nSpikesInProcessedElectrode >= nSpikesThr*mappingDF[mappingDF['electrode']==spikeRefElectrodeNo]['nSpikes'].to_numpy()[0]:
                                if use_hpfAvg:
                                    #path2my_mySpikeDf_withSamples = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithSamples_hpf_DF.csv'
                                    path2my_mySpikeDf_withSamples = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithSamples_DF.csv'
                                else:
                                    path2my_mySpikeDf_withSamples = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithSamples_DF.csv'
                                if not(os.path.isfile(path2my_mySpikeDf_withSamples)):
                                    timeSeriesDf = createTimeSeriesDf(filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, tVec, fs)
                                else:
                                    timeSeriesDf = None
                                
                                print(f'starting processElectrode with spikeRef = {spikeRefElectrodeNo} and processed = {processedElectrodeNo}; electrode {ei} out of {1+len(electrodes_on_branch)}; branch {bi} out of {len(branchs[ni])}; Neuron {ni} out of {len(Neurons)}')
                                print(f'electrode {processedElectrodeNo} has {nSpikesInProcessedElectrode} spikes over {tVec[-1]-tVec[0]} sec')
                                processElectrode(None, filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, timeShiftVsRef, timeSeriesDf, spikes_h5_DF, fs, processElectrode_enableFigures and (ei>0 or (ei==0 and bi==0)), trajectoryOnly, Id_str, use_hpfAvg, process_hpf, processWindow)
                            else:
                                print(f'electrode {processedElectrodeNo} has only {nSpikesInProcessedElectrode} spikes')
                    if saveImages:
                        dst0.save(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_optimizedPatterns.png')
                        print('saved optimizedPattern image')
                        dst1.save(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_corrsOptimalPattern.png')
                        print('saved _corrsOptimalPattern image')
                        dst2.save(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_corrsOptimalPatternHeatmap.png')
                        print('saved _corrsOptimalPatternHeatmap image')
                        dst3.save(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_corrsOptimalPatternHeatmap_extreame.png')
                        print('saved _corrsOptimalPatternHeatmap_extreame image')
                        dst4.save(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_corrsOptimalPatternMaxMarkers.png')
                        print('saved _corrsOptimalPatternMaxMarkers image')
                        for rollingValue in mvAvg_values:
                            dst_rollingDict[rollingValue].save(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_movingAvg_' + rollingValue + '_corrsOptimalPatternMaxMarkers.png')
                            dst_rollingSpatialDict[rollingValue].save(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_movingAvg_' + rollingValue + '_spatial_corrsOptimalPatternMaxMarkers.png')
                            print('saved _corrsOptimalPatternMaxMarkers image for rolling' + rollingValue)
                else:
                    trajectoryOnly = False
                    timeShiftVsRefList = preDefinedTimeShiftVsRefList
                    mySpikeDf = None
                    electrodeProcessList = [NeuronElectrodeNo] + electrodes_on_branch
                
                    if not save_mySpikes:
                        electrode_timeshift_list = [(electrode, timeshift) for electrode, timeshift in zip(electrodeProcessList[1:], timeShiftVsRefList[1:])]
                        electrode_timeshift_list = np.random.permutation(electrode_timeshift_list).tolist()
                        electrodeProcessList = [NeuronElectrodeNo] + [e[0] for e in electrode_timeshift_list]
                        electrodeProcessList = [int(e) for e in electrodeProcessList]
                        timeShiftVsRefList = [timeShiftVsRefList[0]] + [e[1] for e in electrode_timeshift_list]
                    
                    for ei, branchElectrodeNo, timeShiftVsRef in zip(np.arange(1+len(electrodes_on_branch)).tolist(), electrodeProcessList, timeShiftVsRefList):
                        processedElectrodeNo = branchElectrodeNo
                        
                        if not save_mySpikes:
                            path2OptimalPatternIamge = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_optimizedPattern.png'
                            #path2OptimalPatternIamge = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPatternHeatmap_extreame.png'
                            if ei > 0 and os.path.isfile(path2OptimalPatternIamge):
                                print(f'skipping electrode {processedElectrodeNo}')
                                continue
                        
                        
                        if processType == 'processStimuli':
                            Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
                        elif processType == 'processNeuronAndAllElectrodes':
                            Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
                        elif processType == 'processNeuronsAndBranches':
                            Id_str = f'(c,n,b,e)={(int(configuration), Neuron, branch, processedElectrodeNo)}'
                        
                        nSpikesInProcessedElectrode = mappingDF[mappingDF['electrode']==processedElectrodeNo]['nSpikes'].to_numpy()[0]
                        if nSpikesInProcessedElectrode >= nSpikesThr*mappingDF[mappingDF['electrode']==spikeRefElectrodeNo]['nSpikes'].to_numpy()[0]:
                            if use_hpfAvg:
                                #path2my_mySpikeDf_withSamples = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithSamples_hpf_DF.csv'
                                path2my_mySpikeDf_withSamples = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithSamples_DF.csv'
                            else:
                                path2my_mySpikeDf_withSamples = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithSamples_DF.csv'
                            if not(os.path.isfile(path2my_mySpikeDf_withSamples)):
                                timeSeriesDf = createTimeSeriesDf(filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, tVec, fs)
                            else:
                                timeSeriesDf = None
                            print(f'starting processElectrode with spikeRef = {spikeRefElectrodeNo} and processed = {processedElectrodeNo}; electrode {ei} out of {1+len(electrodes_on_branch)}; branch {bi} out of {len(branchs[ni])}; Neuron {ni} out of {len(Neurons)}')
                            print(f'electrode {processedElectrodeNo} has {nSpikesInProcessedElectrode} spikes over {tVec[-1]-tVec[0]} sec')
                            if save_mySpikes:
                                mySpikeDf, _ = processElectrode(mySpikeDf, filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, timeShiftVsRef, timeSeriesDf, spikes_h5_DF, fs, processElectrode_enableFigures and (ei>0 or (ei==0 and bi==0)), trajectoryOnly, Id_str, use_hpfAvg, process_hpf, processWindow)
                            else:
                                path2myUpdatedSpikeFile = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_refElectrode_' + str(spikeRefElectrodeNo)+ '_electrode_' + str(processedElectrodeNo)  + '_mySpikesAll_DF.csv'
                                path2myRefSpikeFile = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_refElectrode_' + str(spikeRefElectrodeNo)+ '_electrode_' + str(processedElectrodeNo)  + '_myRefSpikesAll_DF.csv'
                                if ei == 0:
                                    if not(os.path.isfile(path2myRefSpikeFile)) or not(os.path.isfile(path2OptimalPatternIamge)):
                                        mySpikeDf, _ = processElectrode(mySpikeDf, filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, timeShiftVsRef, timeSeriesDf, spikes_h5_DF, fs, processElectrode_enableFigures and (ei>0 or (ei==0 and bi==0)), trajectoryOnly, Id_str, use_hpfAvg, process_hpf, processWindow)                    
                                        mySpikeDf.to_csv(path2myRefSpikeFile)
                                    else:
                                        mySpikeDf = pd.read_csv(path2myRefSpikeFile)
                                else:
                                    processElectrode(mySpikeDf, filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, timeShiftVsRef, timeSeriesDf, spikes_h5_DF, fs, processElectrode_enableFigures and (ei>0 or (ei==0 and bi==0)), trajectoryOnly, Id_str, use_hpfAvg, process_hpf, processWindow)                            
                                    #tmp, _ = processElectrode(mySpikeDf, filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, timeShiftVsRef, timeSeriesDf, spikes_h5_DF, fs, processElectrode_enableFigures and (ei>0 or (ei==0 and bi==0)), trajectoryOnly, Id_str, use_hpfAvg, process_hpf, processWindow)                            
                                    #tmp.to_csv(path2myUpdatedSpikeFile)
                                
                                
                        else:
                            print(f'electrode {processedElectrodeNo} has only {nSpikesInProcessedElectrode} spikes')
                    
                    
                    if False:
                        #startOfRecordingsIndices = np.concatenate((np.zeros(1),np.where(timeSeriesDf['time'].diff() > 1/fs + 1e-6)[0])).astype(int)
                        startOfRecordingsIndices = np.concatenate((np.zeros(1),np.where(pd.DataFrame(data=tVec,columns=['time'])['time'].diff() > 1/fs + 1e-6)[0])).astype(int)
                        stopOfRecordingsIndices = np.concatenate((startOfRecordingsIndices[1:], np.array([len(tVec)]))).astype(int)
                        
                        # print soma and branch 1:
                        NeuronChannelNo = mappingDF[mappingDF['electrode'] == spikeRefElectrodeNo]['channel'].to_numpy()[0].astype(int)
                        timesOfSpikesInChannel = spikes_h5_DF[spikes_h5_DF['channel'] == NeuronChannelNo]['time'].to_numpy()
                        for i,startIdx,stopIdx in zip(np.arange(startOfRecordingsIndices.shape[0]), startOfRecordingsIndices, stopOfRecordingsIndices):
                            
                            singleTrigStartTime = timeSeriesDf.loc[startIdx,'time']
                            singleTrigStopTime = timeSeriesDf.loc[stopIdx-1,'time']
                            
                            if (timesOfSpikesInChannel[np.logical_and(timesOfSpikesInChannel >= singleTrigStartTime, timesOfSpikesInChannel <= singleTrigStopTime)]).shape[0] > 0:
                                # spike was recorded in this channel
                                neuron_and_branch = ['e '+ str(NeuronElectrodeNo)] + ['e '+ str(ElectrodeNo) for ElectrodeNo in electrodes_on_branch]
                                singleTrig = timeSeriesDf.loc[startIdx:stopIdx-1,neuron_and_branch]
                                
                                singleTrigIdx = timeSeriesDf.loc[startIdx,'trigIdx']
                                df = mySpikeDf[mySpikeDf['trigIdx'] == singleTrigIdx]
                                printSpikeWithFeatures(singleTrig, singleTrigStartTime, df.reset_index(), fs, f'~toas@e {spikeRefElectrodeNo}')
                    
                    
                    
                    path2myUpdatedSpikeFile = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(spikeRefElectrodeNo) + 'branch' + str(branch) + '_mySpikesAll_DF.csv'
                    if save_mySpikes:
                        mySpikeDf.to_csv(path2myUpdatedSpikeFile)
                    
                    if enableSummaryPlots:
                        fig = plt.subplots(nrows=2, ncols=2, constrained_layout=False, figsize=(axx*1*4,ab*2*3), sharex=False, sharey=False)
                        plt.suptitle('Time-of-arrival jitter')
                        xlim = 500
                        minor_ticks_top=np.linspace(-xlim,xlim,21)
                        plt.subplot(2,2,1)
                        listOfColors = ['blue', 'orange', 'green', 'black', 'red']
                        for branchElectrodeNo, color in zip(electrodes_on_branch, listOfColors):
                            #if processType == 'processStimuli':
                            branchElectrodeDistFromRef = -1
                            #else:
                            #    refElectrode_x = metrics_data_Axon[metrics_data_Axon['electrode']==spikeRefElectrodeNo]['x Position [µm]'].to_numpy()[0]
                            #    refElectrode_y = metrics_data_Axon[metrics_data_Axon['electrode']==spikeRefElectrodeNo]['y Position [µm]'].to_numpy()[0]
                            #    branchElectrode_x = metrics_data_Axon[metrics_data_Axon['electrode']==branchElectrodeNo]['x Position [µm]'].to_numpy()[0]
                            #    branchElectrode_y = metrics_data_Axon[metrics_data_Axon['electrode']==branchElectrodeNo]['y Position [µm]'].to_numpy()[0]
                            #    branchElectrodeDistFromRef = 1e-6*np.sqrt(np.power(branchElectrode_x-refElectrode_x, 2) + np.power(branchElectrode_y-refElectrode_y, 2)) # m
                            
                            processedElectrodeNo = branchElectrodeNo
                            if processType == 'processStimuli':
                                Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
                            else:
                                Id_str = f'(c,n,b,e)={(int(configuration), Neuron, branch, processedElectrodeNo)}'
                            SNR_std = mySpikeDf[f'~SNR_std@e {processedElectrodeNo}'].to_numpy()[0]
                            deltaTimeFromRef = mySpikeDf[f'~dtoas@e {processedElectrodeNo}']
                            medianDelay = np.median(deltaTimeFromRef) # sec
                            plt.hist(deltaTimeFromRef/1e-6 - medianDelay/1e-6,bins=100,density=True,color=color,log=False,histtype='step',linewidth=1,cumulative=False,label=Id_str + f'; std={str(round(deltaTimeFromRef.std()/1e-6,1))}us; dist={str(round(branchElectrodeDistFromRef/1e-6, 2))}um')
                        #plt.xlabel('us')
                        plt.xlim([-xlim, xlim])
                        plt.xticks(minor_ticks_top,minor=True)
                        plt.grid(which='both', axis='both')
                        plt.legend()
                        
                        plt.subplot(2,2,2)
                        for branchElectrodeNo, color in zip(electrodes_on_branch, listOfColors):
                            processedElectrodeNo = branchElectrodeNo
                            if processType == 'processStimuli':
                                Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
                            else:
                                Id_str = f'(c,n,b,e)={(int(configuration), Neuron, branch, processedElectrodeNo)}'
                            SNR_std = mySpikeDf[f'~SNR_std@e {processedElectrodeNo}'].to_numpy()[0]
                            deltaTimeFromRef = mySpikeDf[f'~dtoas@e {processedElectrodeNo}']
                            medianDelay = np.median(deltaTimeFromRef) # sec
                            #deltaTimeFromRefMin, deltaTimeFromRefMax = -150e-6+medianDelay, 150e-6+medianDelay
                            #deltaTimeFromRefCropped = deltaTimeFromRef[np.logical_and(deltaTimeFromRef > deltaTimeFromRefMin, deltaTimeFromRef < deltaTimeFromRefMax)]
                            plt.hist(deltaTimeFromRef/1e-6 - medianDelay/1e-6,bins=100,density=True,color=color,log=False,histtype='step',linewidth=1,cumulative=True,label=Id_str + f'CDF; std={str(round(deltaTimeFromRef.std()/1e-6,1))}us; dist={str(round(branchElectrodeDistFromRef/1e-6, 2))}um')
                        #plt.xlabel('us')
                        plt.xlim([-xlim, xlim])
                        plt.xticks(minor_ticks_top,minor=True)
                        plt.grid(which='both', axis='both')
                        plt.legend()
                        
                        plt.subplot(2,2,3)
                        for branchElectrodeNo, color in zip(electrodes_on_branch, listOfColors):
                            processedElectrodeNo = branchElectrodeNo
                            if processType == 'processStimuli':
                                Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
                            else:
                                Id_str = f'(c,n,b,e)={(int(configuration), Neuron, branch, processedElectrodeNo)}'
                            SNR_std = mySpikeDf[f'~SNR_std@e {processedElectrodeNo}'].to_numpy()[0]
                            SNR_db = mySpikeDf[f'~SNR_db@e {processedElectrodeNo}'].to_numpy()[0]
                            deltaTimeFromRef = mySpikeDf[f'~dtoas@e {processedElectrodeNo}']
                            medianDelay = np.median(deltaTimeFromRef) # sec
                            deltaTimeFromRefMin, deltaTimeFromRefMax = np.quantile(deltaTimeFromRef, 15/100), np.quantile(deltaTimeFromRef, 85/100)
                            deltaTimeFromRefCropped = deltaTimeFromRef[np.logical_and(deltaTimeFromRef > deltaTimeFromRefMin, deltaTimeFromRef < deltaTimeFromRefMax)]
                            #plt.hist(deltaTimeFromRefCropped/1e-6 - medianDelay/1e-6,bins=50,density=True,color=color,log=False,histtype='step',linewidth=1,cumulative=False,label=f'70% of data; electrode {processedElectrodeNo}; std={str(round(deltaTimeFromRefCropped.std()/1e-6,1))}us')
                            plt.hist(SNR_std*np.random.randn(int(1e6))/1e-6,bins=100,linestyle='dashed', density=True,color=color,log=False,histtype='step',linewidth=1,cumulative=False,label=Id_str + f'Theoretic TOAstd={str(round(SNR_std/1e-6, 1))}us; SNR={str(round(SNR_db, 2))}db')
                        plt.xlabel('us')
                        plt.grid(which='both', axis='both')
                        plt.xlim([-xlim, xlim])
                        plt.xticks(minor_ticks_top,minor=True)
                        plt.legend()
                        
                        
                        #fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*2*1), sharex=True, sharey=False)
                        plt.subplot(2,2,4)
                        for branchElectrodeNo, color in zip(electrodes_on_branch, listOfColors):
                            #if processType == 'processStimuli':
                            branchElectrodeDistFromRef = -1
                            #else:
                            #    refElectrode_x = metrics_data_Axon[metrics_data_Axon['electrode']==spikeRefElectrodeNo]['x Position [µm]'].to_numpy()[0]
                            #    refElectrode_y = metrics_data_Axon[metrics_data_Axon['electrode']==spikeRefElectrodeNo]['y Position [µm]'].to_numpy()[0]
                            #    branchElectrode_x = metrics_data_Axon[metrics_data_Axon['electrode']==branchElectrodeNo]['x Position [µm]'].to_numpy()[0]
                            #    branchElectrode_y = metrics_data_Axon[metrics_data_Axon['electrode']==branchElectrodeNo]['y Position [µm]'].to_numpy()[0]
                            #    branchElectrodeDistFromRef = 1e-6*np.sqrt(np.power(branchElectrode_x-refElectrode_x, 2) + np.power(branchElectrode_y-refElectrode_y, 2)) # m
                            
                            processedElectrodeNo = branchElectrodeNo
                            if processType == 'processStimuli':
                                Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
                            else:
                                Id_str = f'(c,n,b,e)={(int(configuration), Neuron, branch, processedElectrodeNo)}'
                            fa = mySpikeDf[f'~Pfa@e {processedElectrodeNo}']
                            plt.hist(fa,bins=50,linestyle='dashed', density=True,color=color,log=False,histtype='step',linewidth=1,cumulative=True,label=Id_str + f'dist={str(round(branchElectrodeDistFromRef/1e-6, 2))}um')
                        plt.xlabel('P(noise has match-filter score of detection)')
                        plt.legend()
                        plt.grid()
                        plt.show()
exit()
        
if use_hpfAvg:    
    path2mySpikeFile = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(spikeRefElectrodeNo) + '_mySpikes_hpf_DF.csv'
else:
    path2mySpikeFile = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(spikeRefElectrodeNo) + '_mySpikes_DF.csv'
if os.path.isfile(path2mySpikeFile):
    mySpikeDf = pd.read_csv(path2mySpikeFile)
    nrows=1
    fig, axs = plt.subplots(nrows=nrows, ncols=1, constrained_layout=False, figsize=(axx*1,ab*nrows), sharex=True, sharey=False)
    plt.hist(np.mod(mySpikeDf['time'], 1/fs)/(1e-6),bins=25,density=True,log=False,histtype='step',linewidth=3,cumulative=True,label='peak residual CDF')
    plt.xlabel('usec')
    plt.grid()
    plt.legend()
    plt.show()
                    
                
                
                
                
            
