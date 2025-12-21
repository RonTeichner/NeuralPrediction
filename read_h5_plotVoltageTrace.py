
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm
import os
import h5py
import matplotlib.pyplot as plt


EXPERIMENTS_FILES = {
    "19233_Oct23_2023": "data.raw.h5",
    "21621_Jan2_2024_A__30": "Trace_20240117_15_36_00_Neuron30_spont.raw.h5",
    "21621_Jan2_2024_A__29": "Trace_20240118_07_55_33_Neuron29_spont.raw.h5",
    "21621_Jan2_2024_A__12": "Trace_20240118_12_28_14_Neuron12_spont.raw.h5",
    "21621_Jan2_2024__19": "Trace_20240116_08_49_20_Neuron19_Spont.raw.h5",
    "21621_Jan2_2024__14": "Trace_20240116_11_54_15_Neuron14_Spont.raw.h5",
    "21621_Jan2_2024__23": "Trace_20240116_15_13_47_Neuron23_spont.raw.h5",
    "21621_Jan2_2024__9": "Trace_20240116_18_17_19_Neuron9_spont.raw.h5",
    "19312_Jan28_2024__30": "Trace_20240212_06_45_04_Neuron30_spont.raw.h5",
    "19312_Jan28_2024__19": "Trace_20240212_09_56_43_Neuron19_spont.raw.h5",
    "19312_Jan28_2024__15": "Trace_20240212_13_02_47_Neuron15_spont.raw.h5",
    "19312_Jan28_2024__2": "Trace_20240212_16_17_56_Neuron2_spont.raw.h5",
    "21645_Jan28_2024__3": "Trace_20240207_09_02_44_Neuron3_spont.raw.h5",
    "21645_Jan28_2024__18": "Trace_20240207_14_03_13_Neuron18_spont.raw.h5",
    "21672_Jan2_2024__8": "Trace_20240123_06_42_20_Neuron8_spont.raw.h5",
    "21672_Jan2_2024__7": "Trace_20240123_11_25_43_Neuron7_spont.raw.h5",
    "21672_Jan2_2024__21": "Trace_20240123_19_58_23_Neuron21_spont.raw.h5",
    "21672_Jan2_2024__12": "Trace_20240124_06_16_15_Neuron12_spont.raw.h5"
}

H5_DATA_HEADER = "data_store/data0000"
H5_SETTINGS_GAIN = "/settings/gain"
H5_SETTINGS_HPF = "/settings/hpf"
H5_SETTINGS_LSB = "/settings/lsb"
H5_SETTINGS_SAMPLING = "/settings/sampling"
H5_SETTINGS_SPIKES_THRESHOLD = "/settings/spike_threshold"
H5_CHANNELS_KEY = "/groups/routed/channels"
H5_TRIGGER_CHANNELS_KEY = "/groups/routed/trigger_channels"
H5_FRAME_NOS_KEY = "/groups/routed/frame_nos"
H5_RAW_KEY = "/groups/routed/raw"
H5_MAPPING_KEY = "/settings/mapping"
H5_SPIKES_KEY = "/spikes"
H5_START_TIME_KEY = "/start_time"
H5_STOP_TIME_KEY = "/stop_time"

BASE_PATH = "/Users/ron.teichner/Data/MXBIO/Data"
EXPERIMENTS_DIR_PATH = BASE_PATH + "/raw_data"
DATASET_DIR_PATH = os.path.abspath(os.path.join(BASE_PATH, "data/spikes_and_metrics_dfs"))
#DATASET_FILE = "/dataset_somata.csv"
DATASET_FILE = pd.read_csv("/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/AmitShmidov/regressions_comparisons/spikes_and_metrics_dfs/dataset_somata.csv")
OUTPUT_DIR = "/Users/ron.teichner/Data/MXBIO/Data/recordings"


def create_csv_files(experiment, directory):  
    
    
    filename = f"{EXPERIMENTS_DIR_PATH}/{experiment}/{EXPERIMENTS_FILES[experiment]}"
    
    #### mapping ####
    nConfigurations = 1
    configurations = [str(i).zfill(4) for i in range(nConfigurations)]
    configuration = configurations[0]
    
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
    path2spikeFile = f"{directory}/spikes_h5_DF.csv"
    if os.path.isfile(path2spikeFile):
        spikes_h5_DF = pd.read_csv(path2spikeFile)
    else:
        with h5py.File(filename, mode='r') as h5f:
            for rawPointer in rawPointers:
                rawData = np.array(h5f[rawPointer][:])
                print(rawPointer + f' is of shape {rawData.shape} and represents all spikes metadata')
                spikes_h5_DF = pd.DataFrame(data=rawData, columns=['time', 'channel', 'amplitude'])
                spikes_h5_DF.loc[:,'time'] = np.asarray([t[0]/fs for t in list(rawData)])
                
            spikes_h5_DF.to_csv(path2spikeFile)    
    
    rawPointers = ['/data_store/data' + configuration + '/groups/routed/channels']
    with h5py.File(filename, mode='r') as h5f:
        channels = np.array(h5f[rawPointers[0]][:])
    
    
    mapping = '/data_store/data' + configuration + '/settings/mapping'
    mapping_path = f"{directory}/mapping.csv"
    if not os.path.isfile(mapping_path):
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
            mappingDF.to_csv(mapping_path)
    
def createTimeSeriesDf(spikeRefElectrodeNo, processedElectrodeNo, directory):
    #directory = f"/Users/ron.teichner/Data/MXBIO/Data/recordings/{experiment}"
    mapping_path = f"{directory}/mapping.csv"
    path2spikeFile = f"{directory}/spikes_h5_DF.csv"
    
    
    filename = f"{EXPERIMENTS_DIR_PATH}/{experiment}/{EXPERIMENTS_FILES[experiment]}"
    nConfigurations = 1
    configurations = [str(i).zfill(4) for i in range(nConfigurations)]
    configuration = configurations[0]
    
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
        
    rawPointers = ['/data_store/data' + configuration + '/groups/routed/frame_nos']
    with h5py.File(filename, mode='r') as h5f:
        for rawPointer in rawPointers:
            rawData = np.array(h5f[rawPointer][:])
            print(rawPointer + f' is of shape {rawData.shape} - frame numbers from which data was recorded')
            tVec = rawData/fs # sec
    
    
    
    
    
    rawPointers = ['/data_store/data' + configuration + '/groups/routed/channels']
    with h5py.File(filename, mode='r') as h5f:
        channels = np.array(h5f[rawPointers[0]][:])
    
    mappingDF = pd.read_csv(mapping_path)
    spikes_h5_DF = pd.read_csv(path2spikeFile)
    
    if processedElectrodeNo > -1:
        processedElectrodeIndexInRawRecordings = mappingDF[mappingDF['electrode'] == processedElectrodeNo]['idxInRawData'].to_numpy()[0].astype(int)
    
    spikeRefChannelNo = mappingDF[mappingDF['electrode'] == spikeRefElectrodeNo]['channel'].to_numpy()[0].astype(int)
    firstRefSpikeTime = spikes_h5_DF[spikes_h5_DF['channel']==spikeRefChannelNo]['time'].min()
    lastRefSpikeTime = spikes_h5_DF[spikes_h5_DF['channel'] == spikeRefChannelNo]['time'].max()

    settingPointers = ['/data_store/data' + configuration + '/settings/gain', '/data_store/data' + configuration + '/settings/hpf', '/data_store/data' + configuration + '/settings/lsb', '/data_store/data' + configuration + '/settings/sampling', '/data_store/data' + configuration + '/settings/spike_threshold']
    rawPointers = ['/data_store/data' + configuration + '/groups/routed/raw']
    
    if True:
        
        with h5py.File(filename, "r") as h5f:
            ds = h5f[rawPointers[0]]
            print("Dataset dtype:", ds.dtype)
            print("Dataset shape:", ds.shape)
            print("Dataset ndim :", ds.ndim)

    
    with h5py.File(filename, mode='r') as h5f:
        lsb = np.array(h5f[settingPointers[2]][:])[0]/1000 # volt
        fs = np.array(h5f[settingPointers[3]][:])[0] # hz
        if processedElectrodeNo > -1:
            rawData = np.array(h5f[rawPointers[0]][processedElectrodeIndexInRawRecordings])
        else:
            rawData = np.array(h5f[rawPointers[0]][:])
        rawRecordings = lsb*rawData.astype(float)

    #indices = np.logical_and(tVec >= firstRefSpikeTime, tVec <= lastRefSpikeTime)
    #tVec = tVec[indices]
    #rawRecordings = rawRecordings[indices]
    
    startOfRecordingsIndices = np.concatenate((np.zeros(1),np.where(pd.DataFrame(data=tVec,columns=['time'])['time'].diff() > 1/fs + 1e-6)[0])).astype(int)
    stopOfRecordingsIndices = np.concatenate((startOfRecordingsIndices[1:], np.array([len(tVec)]))).astype(int)
    
    
    
    if processedElectrodeNo > -1:
        timeSeriesDfFeatures = ['time','trigIdx'] + ['e ' + str(processedElectrodeNo)]
        timeSeriesDf = pd.DataFrame(columns=timeSeriesDfFeatures, data=np.concatenate((tVec[:,None], np.zeros((tVec.shape[0],1)), rawRecordings[:, None]), axis=1))    
    else:
        timeSeriesDfFeatures = ['time','trigIdx'] + ['e ' + str(elctrodeNo) for elctrodeNo in mappingDF['electrode'].astype(int).tolist()]
        timeSeriesDf = pd.DataFrame(columns=timeSeriesDfFeatures, data=np.concatenate((tVec[:,None], np.zeros((tVec.shape[0],1)), rawRecordings.transpose()), axis=1))    
    
    for i,startIdx,stopIdx in zip(np.arange(startOfRecordingsIndices.shape[0]), startOfRecordingsIndices, stopOfRecordingsIndices):
        timeSeriesDf.loc[startIdx:stopIdx-1,'trigIdx'] = i
    return timeSeriesDf    

def printSpikeWithFeatures(singleTrig, singleTrigStartTime, spikes_h5_DF_singleChannel, AmitDatasetExpChannel, fs, peakTimeColumnName, nSecondsToPlot=None):
    
    spikesMinimalDistance = 1*1e-3 # sec
    axx, ab = 16/3, 9/3
    tVec = np.arange(len(singleTrig))/fs
    
    if nSecondsToPlot is None:
        nSecondsToPlotWasNone = True
        nSecondsToPlot = tVec[-2]
    else:
        nSecondsToPlotWasNone = False
    maxIdx2Plot = np.where(tVec > nSecondsToPlot)[0][0]
    
    nrows = singleTrig.shape[1]
    fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=False, figsize=(axx*1*10, ab*2*nrows), sharex=True, sharey=False)
    
    ax = axs[0]
        
    ax.plot(tVec[:maxIdx2Plot]/1e-3, singleTrig.iloc[:maxIdx2Plot,0]/1e-6, linewidth=0.5)
    
    # Add all vertical lines at once
    spike_times = spikes_h5_DF_singleChannel['time'].to_numpy()
    spike_amplitudes = spikes_h5_DF_singleChannel['amplitude'].to_numpy()
    
    

    voltage_values = singleTrig.iloc[:,0].to_numpy()
    
    # Spike times you want to sample at
    spike_times = spikes_h5_DF_singleChannel['time'].to_numpy()
    
    # Find indices in tVec that are closest to spike_times
    spike_indices = np.searchsorted(tVec, spike_times)
    
    # Make sure indices are within bounds
    spike_indices = np.clip(spike_indices, 0, len(voltage_values) - 1)
    
    # Get voltage values at spike times
    spike_voltages = voltage_values[spike_indices]
    
    if nSecondsToPlotWasNone:
        nSecondsToPlot = spike_times[-2]
    maxIdx = np.where(spike_times > nSecondsToPlot)[0][0]
    
    bx = axs[1]    
    #ax.plot(spike_times/1e-3, spike_amplitudes/1e-6, 'r.')
    # Plot all vertical lines at once
    bx.vlines(spike_times[:maxIdx]/1e-3, 
              ymin=spike_voltages[:maxIdx]/1e-6, 
              ymax=(spike_voltages - spike_amplitudes)[:maxIdx]/1e-6,
              colors='red', linewidth=2, alpha=0.7, label='Spike amplitudes')
    
    ##########
    
    
    # Add all vertical lines at once
    spike_times = AmitDatasetExpChannel['time'].to_numpy()
    spike_amplitudes = AmitDatasetExpChannel['amplitude'].to_numpy()
    
    

    voltage_values = singleTrig.iloc[:,0].to_numpy()
    
    # Spike times you want to sample at
    spike_times = AmitDatasetExpChannel['time'].to_numpy()
    
    # Find indices in tVec that are closest to spike_times
    spike_indices = np.searchsorted(tVec, spike_times)
    
    # Make sure indices are within bounds
    spike_indices = np.clip(spike_indices, 0, len(voltage_values) - 1)
    
    # Get voltage values at spike times
    spike_voltages = voltage_values[spike_indices]
    
    maxIdx = np.where(spike_times > nSecondsToPlot)[0][0]
        
    #ax.plot(spike_times/1e-3, spike_amplitudes/1e-6, 'r.')
    # Plot all vertical lines at once
    bx.vlines(spike_times[:maxIdx]/1e-3, 
              ymin=spike_voltages[:maxIdx]/1e-6, 
              ymax=(spike_voltages - spike_amplitudes)[:maxIdx]/1e-6,
              colors='green', linestyle='dashed', linewidth=2, alpha=0.7, label='Spike amplitudes')
    ##########
    
    ax.set_xlabel('Time (ms)', fontsize=30)
    ax.set_ylabel('Voltage (µV)', fontsize=30)
    ax.legend()
    ax.grid(True)
    
    #singleTrigIdx = df['trigIdx'].to_numpy()[-1]
    ax.set_title(f'Soma; ' + 'electrode ' + singleTrig.columns[0], fontsize=36)
    #ax.legend(fontsize=36)
    ax.tick_params(labelsize=36)
    fig.tight_layout()
    
    for i in range(1,nrows):
        bx = axs[i]
        bx.plot(tVec/1e-3, singleTrig.iloc[:,i], linewidth=5)
        bx.set_title('electrode ' + singleTrig.columns[i], fontsize=36)
        bx.tick_params(labelsize=36)
        if i == nrows-1:
            bx.set_xlabel('msec', fontsize=36)
        fig.tight_layout()
    
    plt.savefig(peakTimeColumnName, dpi=400)
    plt.close()

for experiment in EXPERIMENTS_FILES.keys():    
    
    directory = OUTPUT_DIR + f"/{experiment}"
    os.makedirs(directory, exist_ok=True)
    
    mapping_path = f"{directory}/mapping.csv"
    path2spikeFile = f"{directory}/spikes_h5_DF.csv"
    
    
    AmitDatasetExp = DATASET_FILE[DATASET_FILE['experiment']==experiment].copy()
    
    filename = f"{EXPERIMENTS_DIR_PATH}/{experiment}/{EXPERIMENTS_FILES[experiment]}"
    nConfigurations = 1
    configurations = [str(i).zfill(4) for i in range(nConfigurations)]
    configuration = configurations[0]
    
    settingPointers = ['/data_store/data' + configuration + '/settings/gain', '/data_store/data' + configuration + '/settings/hpf', '/data_store/data' + configuration + '/settings/lsb', '/data_store/data' + configuration + '/settings/sampling', '/data_store/data' + configuration + '/settings/spike_threshold']
    rawPointers = ['/data_store/data' + configuration + '/groups/routed/raw']
    with h5py.File(filename, mode='r') as h5f:
        lsb = np.array(h5f[settingPointers[2]][:])[0]/1000 # volt
        fs = np.array(h5f[settingPointers[3]][:])[0] # hz
    
    
    if not os.path.isfile(mapping_path):
        create_csv_files(experiment, directory)
    
    spikes_h5_DF = pd.read_csv(path2spikeFile)
    mappingDF = pd.read_csv(mapping_path)
    
    
    path2metrics_data_xlsx = f"{EXPERIMENTS_DIR_PATH}/{experiment}/metrics_data.xlsx"
    metrics_data_Neuron = pd.read_excel(path2metrics_data_xlsx, sheet_name='Axon - Neuron Level')
    Neurons = metrics_data_Neuron['neuron'].to_numpy().tolist()
    
    for Neuron in Neurons:
        spikeRefElectrodeNo = metrics_data_Neuron[metrics_data_Neuron['neuron']==Neuron]['Electrode Number'].to_numpy()[0]
        png_path = f"{directory}/e_{spikeRefElectrodeNo}.png"
        if not os.path.isfile(png_path):
            timeSeriesDf_path = f"{directory}/timeSeries_Neuron_{Neuron}.csv"
            
            electrodeProcessList = [spikeRefElectrodeNo]
            electrodeProcessList = [int(r) for r in electrodeProcessList]
            processedElectrodeNo = electrodeProcessList[0]
            
            if not os.path.isfile(timeSeriesDf_path):
                timeSeriesDf = createTimeSeriesDf(spikeRefElectrodeNo, processedElectrodeNo, directory)
                #timeSeriesDf.to_csv(timeSeriesDf_path)
            else:
                timeSeriesDf = pd.read_csv(timeSeriesDf_path)
            
            plt.close('all')
            nSecondsToPlot = 60*10 # sec
            # Get spike times for a specific channel
            NeuronChannelNo = mappingDF[mappingDF['electrode'] == spikeRefElectrodeNo]['channel'].to_numpy()[0].astype(int)
            timesOfSpikesInChannel = spikes_h5_DF[spikes_h5_DF['channel'] == NeuronChannelNo]['time'].to_numpy()
            
            # Find recording segments
            startOfRecordingsIndices = np.concatenate((np.zeros(1), np.where(timeSeriesDf['time'].diff() > 1/fs + 1e-6)[0])).astype(int)
            stopOfRecordingsIndices = np.concatenate((startOfRecordingsIndices[1:], np.array([len(timeSeriesDf)]))).astype(int)
            
            AmitDatasetExpChannel = AmitDatasetExp[AmitDatasetExp['channel']==NeuronChannelNo].copy()
            
            # Loop through segments and plot if they contain a spike
            for i, startIdx, stopIdx in zip(np.arange(startOfRecordingsIndices.shape[0]), startOfRecordingsIndices, stopOfRecordingsIndices):
                
                singleTrigStartTime = timeSeriesDf.loc[startIdx, 'time']
                singleTrigStopTime = timeSeriesDf.loc[stopIdx-1, 'time']
                
                # Check if this segment contains a spike
                if (timesOfSpikesInChannel[np.logical_and(timesOfSpikesInChannel >= singleTrigStartTime, timesOfSpikesInChannel <= singleTrigStopTime)]).shape[0] > 0:
                    # Extract the time series for neurons and branches
                    neuron = ['e '+ str(spikeRefElectrodeNo)]# + ['e '+ str(ElectrodeNo) for ElectrodeNo in electrodes_on_branch]
                    singleTrig = timeSeriesDf.loc[startIdx:stopIdx-1, neuron]
                    
                    start_time = timeSeriesDf.loc[startIdx, 'time']
                    spikes_h5_DF_singleChannel = spikes_h5_DF[spikes_h5_DF['channel']==NeuronChannelNo].copy()
                    spikes_h5_DF_singleChannel.loc[:,'time'] = spikes_h5_DF_singleChannel['time'] -  start_time
                    spikes_h5_DF_singleChannel.loc[:,'amplitude'] = spikes_h5_DF_singleChannel['amplitude']*lsb
                    AmitDatasetExpChannel.loc[:,'amplitude'] = AmitDatasetExpChannel['amplitude']*lsb
                    # Call plotting function
                    printSpikeWithFeatures(singleTrig, singleTrigStartTime,  spikes_h5_DF_singleChannel, AmitDatasetExpChannel, fs, png_path, nSecondsToPlot)
        else:
            print(f'{png_path} exists')
            
            