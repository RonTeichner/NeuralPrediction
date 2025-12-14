import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import scipy.io as sio
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import blackman, firwin, freqz, firls, upfirdn, resample_poly, lfilter, convolve, correlate
from scipy import signal
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime
import platform
import re
from scipy import interpolate
import h5py
import os

def numberOutOfStr(r):
    numbersIn_r_list = re.findall(r'\d+', r)
    if len(numbersIn_r_list) == 1:
        number = int(numbersIn_r_list[0])
    elif len(numbersIn_r_list) == 2:
        number = float(numbersIn_r_list[0])
    else:
        assert False
    return number

def activityLoop(ei, lengthElectrodeProcessList, filename, analysisLibrary, nameOf_h5, mySpikeDf, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, tVec, fs, use_hpfAvg, process_hpf, processWindow, spikes_h5_DF, timeShiftVsRef, removeStimuliInterference, stimuliElectrodeNo):
    cstr = 'e ' + str(processedElectrodeNo) + ' hpf'
    #if partialFile and cstr in avgPatternDf.columns:
    #    continue
    Id_str = f'(c,e)={(int(configuration), processedElectrodeNo)}'
    path2my_mySpikeDf_withSamples = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithSamples_DF.csv'
    if not(os.path.isfile(path2my_mySpikeDf_withSamples)):
        timeSeriesDf = createTimeSeriesDf(filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, tVec, fs)
    else:
        timeSeriesDf = None
    print(f'{datetime.datetime.now()}: starting processElectrode with spikeRef = {spikeRefElectrodeNo} and processed = {processedElectrodeNo}; electrode {ei} out of {lengthElectrodeProcessList}')
    avgPatternSeries, _ = processElectrode(mySpikeDf, filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, timeShiftVsRef, timeSeriesDf, spikes_h5_DF, fs, False, True, Id_str, use_hpfAvg, process_hpf, processWindow, removeStimuliInterference, stimuliElectrodeNo)
    return avgPatternSeries

class NeuroDataset(Dataset):
    def __init__(self, observations, transform=None):
        self.observations = observations
        self.transform = transform
        
    def __len__(self):
        # 'Denotes the total number of samples'
        return self.observations.shape[0]
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #sample = {'measurements': self.patientsDf.iloc[range(idx*self.nTime, (idx+1)*self.nTime), range(3, 3+self.nFeatures)].to_numpy(dtype='float32')}
        sample = self.observations[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


def createTimeSeriesDf(filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, tVec, fs):
    #########################
    bands = (0, 300, 700, fs/2,)
    desired = (0, 0, 1, 1)
    numtaps = 25#13
    fir_firls = signal.firls(numtaps, bands, desired, fs=fs)
    sos = signal.tf2sos(b=fir_firls, a=1)
    '''
    fig, axs = plt.subplots(1)
    hs = list()
    ax = axs
    freq, response = signal.freqz(fir_firls)
    hs.append(ax.semilogy(0.5*fs*freq/np.pi, np.abs(response))[0])
    for band, gains in zip(zip(bands[::2], bands[1::2]),
                           zip(desired[::2], desired[1::2])):
        ax.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)
    ax.legend(hs, 'firls', loc='lower center', frameon=False)
    ax.set_xlabel('Frequency (Hz)')
    ax.grid(True)
    ax.set(title='High-pass %d-%d Hz' % bands[2:4], ylabel='Magnitude')
    
    fig.tight_layout()
    plt.show()
    '''
    ########################
    
    rawPointers = ['/data_store/data' + configuration + '/groups/routed/channels']
    with h5py.File(filename, mode='r') as h5f:
        channels = np.array(h5f[rawPointers[0]][:])
    
    if processedElectrodeNo > -1:
        processedElectrodeIndexInRawRecordings = mappingDF[mappingDF['electrode'] == processedElectrodeNo]['idxInRawData'].to_numpy()[0].astype(int)
    
    path2spikeFile = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_spikes_h5_DF.csv'
    spikes_h5_DF = pd.read_csv(path2spikeFile)

    spikeRefChannelNo = mappingDF[mappingDF['electrode'] == spikeRefElectrodeNo]['channel'].to_numpy()[0].astype(int)
    firstRefSpikeTime = spikes_h5_DF[spikes_h5_DF['channel']==spikeRefChannelNo]['time'].min()
    lastRefSpikeTime = spikes_h5_DF[spikes_h5_DF['channel'] == spikeRefChannelNo]['time'].max()

    settingPointers = ['/data_store/data' + configuration + '/settings/gain', '/data_store/data' + configuration + '/settings/hpf', '/data_store/data' + configuration + '/settings/lsb', '/data_store/data' + configuration + '/settings/sampling', '/data_store/data' + configuration + '/settings/spike_threshold']
    rawPointers = ['/data_store/data' + configuration + '/groups/routed/raw']
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
        hpf_rawRecordings = signal.sosfilt(sos, x=rawRecordings)
        hpf_rawRecordings[:len(fir_firls)] = 0
    
    
    if processedElectrodeNo > -1:
        timeSeriesDfFeatures = ['time','trigIdx'] + ['e ' + str(processedElectrodeNo), 'e ' + str(processedElectrodeNo) + ' hpf']
        timeSeriesDf = pd.DataFrame(columns=timeSeriesDfFeatures, data=np.concatenate((tVec[:,None], np.zeros((tVec.shape[0],1)), rawRecordings[:, None], hpf_rawRecordings[:, None]), axis=1))    
    else:
        timeSeriesDfFeatures = ['time','trigIdx'] + ['e ' + str(elctrodeNo) for elctrodeNo in mappingDF['electrode'].astype(int).tolist()]
        timeSeriesDf = pd.DataFrame(columns=timeSeriesDfFeatures, data=np.concatenate((tVec[:,None], np.zeros((tVec.shape[0],1)), rawRecordings.transpose()), axis=1))    
    
    for i,startIdx,stopIdx in zip(np.arange(startOfRecordingsIndices.shape[0]), startOfRecordingsIndices, stopOfRecordingsIndices):
        timeSeriesDf.loc[startIdx:stopIdx-1,'trigIdx'] = i
    return timeSeriesDf

def calcMeasNoiseTOA_var(SNR, N, pattern):
    pattern = pattern - pattern.mean()
    signalEnergyPerSample = np.power(pattern, 2).mean()
    noiseEnergyPerSample = signalEnergyPerSample/SNR
    noiseStd = np.sqrt(noiseEnergyPerSample)
    startIdx = int(N/2-len(pattern)/2)
    
    corrs = np.zeros((1000, N-len(pattern)+1))
    for i in range(corrs.shape[0]):
        noise = noiseStd*np.random.randn(N)
        sig = noise
        sig[startIdx:startIdx+len(pattern)] = noise[startIdx:startIdx+len(pattern)] + pattern 
        corrs[i] = normalizedCorrelator(sig[None,:], pattern)[0]
    
    _, locsCorrs = Corerlator_ANN(torch.zeros((1,5))).polyFitCorr(torch.from_numpy(corrs).unsqueeze(-1).type(torch.float))
    
    
    return locsCorrs.var().detach().cpu().numpy()
    

def estimateSNR(mySpikeDf_withSamples, locsCorrsOptimal, N, patternLength, Id_str, process_hpf, enablePrint):
    # estimate noise energy:
    noises = np.zeros((0))
    for index, row in mySpikeDf_withSamples.iterrows():
        if process_hpf:
            singleNoiseTimeSeries = row.loc['hpf csc=0':f'hpf csc={N-1}']
        else:
            singleNoiseTimeSeries = row.loc['csc=0':f'csc={N-1}']
        singleNoiseTimeSeries = singleNoiseTimeSeries - singleNoiseTimeSeries.mean()
        noises = np.concatenate((noises, singleNoiseTimeSeries))
    meanNoiseEnergyPerSample = np.power(noises, 2).mean()
    
    # estimate signal energy:
    signal_and_noise_energy_perSample = np.zeros(mySpikeDf_withSamples.shape[0])
    signal_energy_perSample = np.zeros(mySpikeDf_withSamples.shape[0])
    sig_n_noises = np.zeros((0))
    for index, row in mySpikeDf_withSamples.iterrows():  
        if process_hpf:
            singleTimeSeries = row.loc['hpf s=0':f'hpf s={N-1}']
        else:
            singleTimeSeries = row.loc['s=0':f's={N-1}']
        startIdx = int(locsCorrsOptimal[index].round().detach().cpu().numpy())
        croppedAroundMatchedPattern = singleTimeSeries[startIdx:startIdx+patternLength]
        noise = np.concatenate((singleTimeSeries[:startIdx], singleTimeSeries[startIdx+patternLength:]))
        croppedAroundMatchedPattern = croppedAroundMatchedPattern - noise.mean()
        sig_n_noises = np.concatenate((sig_n_noises, croppedAroundMatchedPattern))
        
    signal_and_noise_energy_perSample = np.power(sig_n_noises, 2).mean()
    signal_energy_perSample = signal_and_noise_energy_perSample - meanNoiseEnergyPerSample
    
    SNR = signal_energy_perSample/meanNoiseEnergyPerSample
    if enablePrint:
        plt.figure()
        plt.hist(np.power(noises, 2),bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=True,label=f'noise energy CDF')
        plt.hist(np.power(sig_n_noises, 2),bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=True,label=f'signal+noise energy CDF')
        plt.grid()
        plt.legend()
        plt.title(Id_str)
        plt.show()
    
        
    return SNR

class Corerlator_ANN(nn.Module):
  def __init__(self, initPatterns):
        super(Corerlator_ANN, self).__init__()
        self.nInitPatterns, self.nSamplesInPattern = initPatterns.shape
        self.nPatterns = 1#25
        initPatterns = initPatterns.type(torch.float)
        initPatterns = torch.divide(initPatterns-initPatterns.mean(axis=1)[:,None].expand(-1,self.nSamplesInPattern), initPatterns.std(axis=1)[:,None].expand(-1,self.nSamplesInPattern))
        randInit = torch.randn(self.nPatterns-self.nInitPatterns, self.nSamplesInPattern)
        initPatterns = torch.cat((initPatterns, randInit), dim=0)
        
        self.h = nn.parameter.Parameter(initPatterns.type(torch.float), requires_grad=True)
        
  def polyFitCorr(self, corrs):
      P, T, F = corrs.shape
      peaks, locs = torch.zeros((P), device=corrs.device), torch.zeros((P), device=corrs.device)
      maxIndices = corrs.argmax(dim=1)[:,0]
      
      peaks[maxIndices == 0] = corrs[maxIndices == 0,0,0]
      locs[maxIndices == 0] = 0
      peaks[maxIndices == T-1] = corrs[maxIndices == T-1,T-1,0]
      locs[maxIndices == T-1] = T-1
      
      intermediateIndices = torch.logical_and(maxIndices>0, maxIndices<T-1)
      intermediateCorrs = corrs[intermediateIndices]
      intermediateMaxIndices = maxIndices[intermediateIndices].unsqueeze(-1).unsqueeze(-1)
      intermediateMaxIndices = torch.cat((intermediateMaxIndices-1, intermediateMaxIndices, intermediateMaxIndices+1), dim=1)
      threeSamples = torch.gather(input=intermediateCorrs, dim=1, index=intermediateMaxIndices)
      
      # polyfit:
      b = 0.5*(threeSamples[:,2,0] - threeSamples[:,0,0])
      a = 0.5*(threeSamples[:,0,0] + threeSamples[:,2,0] - 2*threeSamples[:,1,0])
      c = threeSamples[:,1,0]
      
      relativeLocs = torch.divide(-b, 2*a)
      locs[intermediateIndices] = relativeLocs + intermediateMaxIndices[:,1,0]
      peaks[intermediateIndices] = torch.multiply(a, torch.pow(relativeLocs, 2)) + torch.multiply(b, relativeLocs) + c
      
      
      return peaks, locs
  
  def correlator(self, x, pattern):
      P, T, F = x.shape
      nPatterns, nSamplesInPattern = pattern.shape
      pattern = torch.divide(pattern - pattern.mean(dim=1).unsqueeze(-1).expand(-1, nSamplesInPattern), pattern.std(dim=1).unsqueeze(-1).expand(-1,nSamplesInPattern))
      patternEnergy = torch.pow(pattern, 2).sum(dim=1)
      corrs = torch.zeros((nPatterns, P, T - nSamplesInPattern + 1, 1), device=x.device)
      for i in range(T - nSamplesInPattern):
          partialSig = x[:, i:i+nSamplesInPattern]
          #partialSig = partialSig - partialSig.mean(dim=1).unsqueeze(-1).expand(-1,nSamplesInPattern,-1)
          #partialSigEnergy = torch.pow(partialSig, 2).sum(dim=1).unsqueeze(-1)
          for p in range(nPatterns):
              #corrs[p,:,i:i+1] = torch.matmul(pattern[p].unsqueeze(0), partialSig)/torch.sqrt(patternEnergy[p].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(P,-1,-1))/torch.sqrt(partialSigEnergy)
              corrs[p,:,i:i+1] = torch.matmul(pattern[p].unsqueeze(0), partialSig)#/torch.sqrt(patternEnergy[p].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(P,-1,-1))/torch.sqrt(partialSigEnergy)
      #corrs = torch.divide(corrs, patternEnergy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, P, T - nSamplesInPattern + 1, 1))
      return corrs
      
      
  def forward(self, x, enablePolyFit=True):
      P, T, F = x.shape
      pattern = self.h
      corrs = self.correlator(x, pattern)
      if enablePolyFit:
          peaks, locs = torch.zeros((self.nPatterns, P), device=corrs.device), torch.zeros((self.nPatterns, P), device=corrs.device)
          for p in range(self.nPatterns):
              peaks[p], locs[p] = self.polyFitCorr(corrs[p])
      else:
          peaks, locs = corrs[:,:,:,0].max(axis=2)
      
      return corrs, peaks, locs 

def chooseInitPatterns(filename, configuration, processedElectrodeNo, mySpikeDf, nInitPatterns, patternLength):
    '''
    path2spikeFile = filename + '_conf_' + configuration + '_spikes_h5_DF.csv'
    spikes_h5_DF = pd.read_csv(path2spikeFile)
    mappingDF = pd.read_csv(filename + '_conf_' + configuration + '_mapping.csv')
    
    settingPointers = ['/data_store/data' + configuration + '/settings/gain', '/data_store/data' + configuration + '/settings/hpf', '/data_store/data' + configuration + '/settings/lsb', '/data_store/data' + configuration + '/settings/sampling', '/data_store/data' + configuration + '/settings/spike_threshold']
    rawPointers = ['/data_store/data' + configuration + '/groups/routed/frame_nos']
    with h5py.File(filename, mode='r') as h5f:
        fs = np.array(h5f[settingPointers[3]][:])[0] # hz
        tVec = np.array(h5f[rawPointers[0]][:])/fs # sec
    
    timeSeriesDf = createTimeSeriesDf(filename, configuration, mappingDF, processedElectrodeNo, tVec, fs)
    
    processedElectrodeChannelNo = mappingDF[mappingDF['electrode'] == processedElectrodeNo]['channel'].to_numpy()[0].astype(int)
    timesOfSpikes = spikes_h5_DF[spikes_h5_DF['channel']==processedElectrodeChannelNo]['time'].tolist()
    j=0
    for i,timeOfSpike in enumerate(timesOfSpikes):
        if timeOfSpike < timeSeriesDf['time'].to_numpy()[0] or timeOfSpike > timeSeriesDf['time'].to_numpy()[-1]:
            continue
        
        indexOfSpike = (timeSeriesDf['time']-timeOfSpike).abs().argmin()
        startIdx = int(np.max([0,indexOfSpike]))
        startIdx = int(np.min([timeSeriesDf.shape[0]-patternLength,startIdx]))
        if j == 0:
            spikePatern = timeSeriesDf['e ' + str(processedElectrodeNo)][startIdx:startIdx+patternLength].to_numpy()
        else:
            spikePatern = spikePatern + timeSeriesDf['e ' + str(processedElectrodeNo)][startIdx:startIdx+patternLength].to_numpy()
        j += 1
    spikePatern = np.divide(spikePatern, j)
    '''
    P, T, F = observations.shape
    
    patientIndices = torch.arange(P)
    maxValues, maxIndices = observations.abs().max(axis=1)
    maxValues, maxIndices = maxValues[:,0], maxIndices[:,0]
    maxValues, maxIndices, patientIndices = maxValues[maxValues.argsort()], maxIndices[maxValues.argsort()], patientIndices[maxValues.argsort()]
    maxValues, maxIndices, patientIndices = maxValues[::int(P/nInitPatterns)], maxIndices[::int(P/nInitPatterns)], patientIndices[::int(P/nInitPatterns)]
    initPatterns = torch.zeros((len(maxValues)+1, patternLength))
    
    meanPattern = observations.mean(axis=0)
    maxValueMean, maxIndexMean = meanPattern.max(axis=0)
    startIdx = int(torch.max(torch.zeros(1),maxIndexMean-int(patternLength/2)))
    startIdx = int(np.min([T-patternLength,startIdx]))
    initPatterns[0] = meanPattern[:,0][startIdx:startIdx+patternLength]
    #initPatterns[1] = torch.from_numpy(spikePatern).type(torch.float)
    
    for i in range(len(maxValues)):
        startIdx = int(torch.max(torch.zeros(1),maxIndices[i]-int(patternLength/2)))
        startIdx = int(np.min([T-patternLength,startIdx]))
        initPatterns[i+1] = observations[patientIndices[i], startIdx:startIdx+patternLength, 0]
    return initPatterns

def patternOptimization(filename, analysisLibrary, nameOf_h5, configuration, processedElectrodeNo, mySpikeDf, N, patternLength, use_hpfAvg, enablePrint, Id_str, process_hpf):
    if process_hpf:
        observations = torch.from_numpy(mySpikeDf.loc[:, 'hpf s=0':f'hpf s={N-1}'].to_numpy()[:,:,None]).type(torch.float)
    else:
        observations = torch.from_numpy(mySpikeDf.loc[:, 's=0':f's={N-1}'].to_numpy()[:,:,None]).type(torch.float)
    P, T, F = observations.shape
    params = {'batch_size': P, 'shuffle': True}
    bs = params['batch_size']
    print(f'batch size = {bs}')
    training_set = NeuroDataset(observations)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    
    #nInitPatterns = 10
    #initPatterns = chooseInitPatterns(filename, configuration, processedElectrodeNo, mySpikeDf, nInitPatterns, patternLength)
    
    hpf_ts_sub_meidan = mySpikeDf.loc[:, 'hpf s=0':f'hpf s={N-1}'] - np.repeat(mySpikeDf.loc[:, 'hpf s=0':f'hpf s={N-1}'].median(axis=1).to_numpy()[:,None], N, 1)
    hpf_avg = hpf_ts_sub_meidan.loc[:, 'hpf s=0':f'hpf s={N-1}'].mean()
    
    fullPattern = hpf_avg#mySpikeDf.loc[:, 'hpf s=0':f'hpf s={N-1}'].mean()
    fullPattern = fullPattern - fullPattern.median()
    maxIdx = fullPattern.abs().argmax()
    startIdx = int(np.max([0,maxIdx-int(patternLength/2)]))
    startIdx = int(np.min([len(fullPattern)-patternLength,startIdx]))
    initPatterns = torch.from_numpy(fullPattern[startIdx:startIdx+patternLength].to_numpy()[None,:]).type(torch.float)
    
    if use_hpfAvg:
        optimalPattern = initPatterns[0].detach().cpu().numpy()
        if enablePrint:
            plt.figure()
            fullPattern_tVec = np.arange(len(fullPattern))/20000
            
            pstartIdx = int(np.max([0,startIdx-int(patternLength)]))
            pstopIdx = np.min([pstartIdx + int(3*patternLength), len(fullPattern)])
            
            plt.plot(fullPattern_tVec[pstartIdx:pstopIdx], fullPattern[pstartIdx:pstopIdx], 'k', linewidth=0.5)
            plt.plot(fullPattern_tVec[startIdx:startIdx+patternLength], optimalPattern, 'r', linewidth=0.5)
            plt.title(Id_str)
            plt.show()
    else:
        corrPlayer = Corerlator_ANN(initPatterns)
        modelParams = corrPlayer.parameters()
        #optimizer = optim.SGD(modelParams, lr=0.01, momentum=0.99, weight_decay=0.000)
        optimizer = optim.Adam(modelParams, lr=0.01)#, weight_decay=0.001)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=300, factor=0.1)#, threshold=0.01, threshold_mode='abs', verbose=True)
        
        totalUpdateSteps = 5000#250
        nEpochs = int(np.ceil(totalUpdateSteps/(observations.shape[0]/params['batch_size'])))
        print(f'nEpochs = {nEpochs}')
        minLR = 1e-5
        
        device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        print('training on device ' + f'{device}')
        corrPlayer.to(device)
        
        lossOnAbsCorrs = False
        bestLossAchieved = 0
        for epoch in range(nEpochs):
            for i_batch,local_batch in enumerate(training_generator):
                # Transfer to GPU
                local_batch = local_batch.to(device)
                optimizer.zero_grad()
                # correlation player:  
                correlations, peaks, locs = corrPlayer(local_batch)
                
                            
                if lossOnAbsCorrs:
                    if epoch == 0 and i_batch == 0:
                        avgPatternLoss = -correlations.abs().sum(axis=2).mean(axis=1)[0].item()
                        optimalPattern = corrPlayer.h[0].detach().cpu().numpy()
                    else:
                        bestPatternIdx = correlations.abs().sum(axis=2).mean(axis=1).argmax(axis=0)[0]
                        bestLoss = -correlations.abs().sum(axis=2).mean(axis=1)[bestPatternIdx][0].item()
                        if bestLoss < bestLossAchieved:
                            bestLossAchieved = bestLoss
                            optimalPattern = corrPlayer.h[bestPatternIdx].detach().cpu().numpy()
                    loss = -correlations.abs().sum(axis=2).mean()
                else:
                    if epoch == 0 and i_batch == 0:
                        avgPatternLoss = -peaks[0].mean().item()
                        
                    
                    bestPatternIdx = peaks.mean(dim=1).argmax()
                    bestLoss = -peaks[bestPatternIdx].mean().item()
                    if bestLoss < bestLossAchieved:
                        bestLossAchieved = bestLoss
                        optimalPattern = corrPlayer.h[bestPatternIdx].detach().cpu().numpy()
                    
                    loss = -peaks.mean()
                    
                loss.backward()
                optimizer.step()  # parameter update
                #scheduler.step(loss)
                if (epoch > 0 and i_batch == 0 and np.mod(epoch, 100)==0) or (epoch == 0 and i_batch == 1):
                    print(f'epoch, batch {epoch},{i_batch}: avgPatternLoss = {str(avgPatternLoss)}; bestLossAchieved = {str(bestLossAchieved)}; loss = {str(loss.item())};')# learning rate = {scheduler._last_lr[-1]}')
                #if scheduler._last_lr[-1] < minLR:
                 #   break
    
    #optimalPattern = corrPlayer.h[peaks.mean(dim=1).argmax()].detach().cpu().numpy()
    optimalPattern = (optimalPattern-optimalPattern.mean())/optimalPattern.std()
    optimalPattern = np.divide(optimalPattern, np.sqrt(np.power(optimalPattern, 2).sum()))
    '''
    # take 95% of the energy:
    optimalPatternEnergy = np.power(optimalPattern, 2).sum()
    energyFactor = 0.98
    breakFlag = False
    for winSize in range(1,1+len(optimalPattern)):
        for startIdx in range(len(optimalPattern)-winSize):
            croppedOptimalPattern = optimalPattern[startIdx:startIdx+winSize]
            croppedOptimalPatternEnergy = np.power(croppedOptimalPattern, 2).sum()
            if croppedOptimalPatternEnergy/optimalPatternEnergy > energyFactor:
                optimalPattern = croppedOptimalPattern
                breakFlag = True
                break
        if breakFlag:
            break
    optimalPattern = np.divide(optimalPattern, np.sqrt(np.power(optimalPattern, 2).sum()))
    '''
    return optimalPattern

    
def normalizedCorrelator(sig, pattern, enablePolyFit=False, enableTypeConvFull=False):
    Corerlator_ANN_inst = Corerlator_ANN(torch.zeros((1,5)))
    x = torch.from_numpy(sig).unsqueeze(-1).type(torch.float)
    P, T, F = x.shape
    
    if enableTypeConvFull:
        sigMean, patternMean = pd.Series(sig[0]).mean(), pd.Series(pattern).mean()
        sig, pattern = sig - sigMean, pattern - patternMean
        sig[np.isnan(sig)] = 0
        pattern[np.isnan(pattern)] = 0
        corrs = correlate(sig[0], pattern, 'full')[None, None, :, None]
        corrs = torch.tensor(corrs, dtype=torch.float)
    else:
        # sig has shape (1,N); pattern has shape (P,)
        
        
        corrs = Corerlator_ANN_inst.correlator(x, torch.from_numpy(pattern).unsqueeze(0).type(torch.float))
    if enablePolyFit:
        peaks, locs = torch.zeros((Corerlator_ANN_inst.nPatterns, P), device=corrs.device), torch.zeros((Corerlator_ANN_inst.nPatterns, P), device=corrs.device)
        for p in range(Corerlator_ANN_inst.nPatterns):
            peaks[p], locs[p] = Corerlator_ANN_inst.polyFitCorr(corrs[p])
        return corrs[0,:,:,0].detach().cpu().numpy(), peaks.detach().cpu().numpy(), locs.detach().cpu().numpy() 
    else:
        return corrs[0,:,:,0].detach().cpu().numpy()
      

def spikeFeaturesExtract(singleTrig, singleTrigStartTime, previousTrigTime, fs, singleTrigIdx, croppedTimesOfSpikesInRefChannel, selfSpikeEstimation):
    
    if selfSpikeEstimation:
        spikesMinimalDistance = 1*1e-3 # sec
        spikeThr = 40
        
        tVec = np.arange(len(singleTrig))/fs
        maxSpikeValue = singleTrig.min() + spikeThr/100*(singleTrig.max()-singleTrig.min())#np.quantile(singleTrig, spikeThr/100)
        peaks, _ = find_peaks(-singleTrig, height=-maxSpikeValue, distance=np.round(spikesMinimalDistance*fs))
    
        f = interpolate.interp1d(tVec, singleTrig)
        nSamplesUpsampled = np.floor(tVec[-1]*(fs*10))
        tVecUpsampled = np.arange(nSamplesUpsampled)/(fs*10)
        singleTrigUpsampled = f(tVecUpsampled)   # use interpolation function returned by `interp1d`
        
        completeFeatures = ['T','v','dt_minus','dt_plus','dv_minus','dv_plus','trigIdx']
        df = pd.DataFrame(columns=["time", "Id", "batch"] + completeFeatures, data=np.zeros((len(peaks), 3+len(completeFeatures))))
        Id, batch = 0,0
        for i,peak in enumerate(peaks):
            coeffs = np.polyfit(tVec[peak-1:peak+2], singleTrig[peak-1:peak+2], deg=2)
            p = np.poly1d(coeffs)
            analyticPeakTime = -coeffs[1]/(2*coeffs[0])
            analyticPeakVal = p(analyticPeakTime)
            
            if i==0:
                T = (analyticPeakTime + singleTrigStartTime) - previousTrigTime
            else:
                T = (analyticPeakTime + singleTrigStartTime) - df.loc[i-1,'time']
            
            peakIdxIntVecUpsampled = np.argmin(np.abs(tVecUpsampled-analyticPeakTime))
            beforeIdxIntVecUpsampled = np.argmin(np.abs(tVecUpsampled-(analyticPeakTime-spikesMinimalDistance)))
            afterIdxIntVecUpsampled = np.argmin(np.abs(tVecUpsampled-(analyticPeakTime+spikesMinimalDistance)))
            v_minus = np.median(singleTrigUpsampled[beforeIdxIntVecUpsampled:peakIdxIntVecUpsampled])
            v_plus = np.median(singleTrigUpsampled[peakIdxIntVecUpsampled:afterIdxIntVecUpsampled])
            delta_v_minus, delta_v_plus = v_minus-analyticPeakVal, v_plus-analyticPeakVal
            t_minus = tVecUpsampled[beforeIdxIntVecUpsampled + np.where(analyticPeakVal+0.9*delta_v_minus < singleTrigUpsampled[beforeIdxIntVecUpsampled:peakIdxIntVecUpsampled])[0][-1]]
            t_plus = tVecUpsampled[peakIdxIntVecUpsampled + np.where(analyticPeakVal+0.9*delta_v_plus < singleTrigUpsampled[peakIdxIntVecUpsampled:afterIdxIntVecUpsampled])[0][0]]
            delta_t_minus, delta_t_plus = analyticPeakTime - t_minus, t_plus - analyticPeakTime
            
            df.iloc[i] = np.array([analyticPeakTime+singleTrigStartTime, Id, batch, T, analyticPeakVal, delta_t_minus, delta_t_plus, delta_v_minus, delta_v_plus, singleTrigIdx], dtype=float)
    else:
        noDetails = True
        completeFeatures = ['v','trigIdx']
        Id, batch = 0,0
        if noDetails:
            df = pd.DataFrame(columns=["time", "Id", "batch"] + completeFeatures, data=np.concatenate((croppedTimesOfSpikesInRefChannel['time'].to_numpy()[:,None], Id*np.ones((croppedTimesOfSpikesInRefChannel.shape[0],1)), batch*np.ones((croppedTimesOfSpikesInRefChannel.shape[0],1)), np.nan*np.ones((croppedTimesOfSpikesInRefChannel.shape[0],1)), singleTrigIdx*np.ones((croppedTimesOfSpikesInRefChannel.shape[0],1))), axis=1))
        else:
            tVec = singleTrigStartTime + np.arange(len(singleTrig))/fs
            df = pd.DataFrame(columns=["time", "Id", "batch"] + completeFeatures, data=np.zeros((croppedTimesOfSpikesInRefChannel.shape[0], 3+len(completeFeatures))))
            for index, row in croppedTimesOfSpikesInRefChannel.reset_index().iterrows():
                print(f'spikeFeaturesExtract: {index} out of {croppedTimesOfSpikesInRefChannel.shape[0]}: {str(round(index/croppedTimesOfSpikesInRefChannel.shape[0]*100,1))}%')
                systemPeakTime = row['time']
                #systemPeakVal = singleTrig[np.argmin(np.abs(tVec-systemPeakTime))]
                systemPeakVal = singleTrig[np.searchsorted(tVec - systemPeakTime, 0, 'left')]
                df.iloc[index] = np.array([systemPeakTime, Id, batch, systemPeakVal, singleTrigIdx], dtype=float)
    
    return df

def printSpikeWithFeatures(singleTrig, singleTrigStartTime, df, fs, peakTimeColumnName):
    spikesMinimalDistance = 1*1e-3 # sec
    axx, ab = 16/3, 9/3
    tVec = np.arange(len(singleTrig))/fs
    nrows = singleTrig.shape[1]
    fig, axs = plt.subplots(nrows=nrows, ncols=1, constrained_layout=False, figsize=(axx*1*10, ab*2*nrows), sharex=True, sharey=False)
    if nrows > 1:
        ax = axs[0]
    else:
        ax = axs
        
    ax.plot(tVec/1e-3, singleTrig.iloc[:,0], linewidth=5)

    for i in range(df.shape[0]):
        #analyticPeakTime = df.loc[i,'time'] - singleTrigStartTime
        analyticPeakTime = df.loc[i,peakTimeColumnName] - singleTrigStartTime
        analyticPeakVal = df.loc[i,'v']
        ax.plot(analyticPeakTime/1e-3, analyticPeakVal, 'ko', linewidth=5, markersize=30, label=r'$v_{j,p}$')
        if 'dv_minus' in df.columns:
            v_minus = analyticPeakVal + df.loc[i,'dv_minus']
            v_plus = analyticPeakVal + df.loc[i,'dv_plus']
            t_minus = analyticPeakTime - df.loc[i,'dt_minus']
            t_plus = analyticPeakTime + df.loc[i,'dt_plus']

            beforeIdx = np.argmin(np.abs(tVec-(analyticPeakTime-spikesMinimalDistance)))
            peakIdx = np.argmin(np.abs(tVec-analyticPeakTime))
            afterIdx = np.argmin(np.abs(tVec-(analyticPeakTime+spikesMinimalDistance)))
            ax.plot(tVec[beforeIdx:peakIdx]/1e-3, v_minus*np.ones(peakIdx-beforeIdx),'--', linewidth=10, markersize=30, label=r'$v_{j,-}$')
            ax.plot(tVec[peakIdx:afterIdx]/1e-3, v_plus*np.ones(afterIdx-peakIdx),'--', linewidth=10, markersize=30, label=r'$v_{j,+}$')
            ax.axvline(t_minus/1e-3, color='k', linewidth=5, markersize=30, linestyle='dashed')
            ax.axvline(t_plus/1e-3, color='k', linewidth=5, markersize=30, linestyle='dashed')
    ax.grid()
    if nrows == 1:
        ax.set_xlabel('msec', fontsize=36)
    singleTrigIdx = df['trigIdx'].to_numpy()[-1]
    ax.set_title(f'Soma; idx={singleTrigIdx} ' + 'electrode ' + singleTrig.columns[0], fontsize=36)
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
    
    plt.show()
    
def processElectrode(mySpikeDf, filename, analysisLibrary, nameOf_h5, configuration, mappingDF, spikeRefElectrodeNo, processedElectrodeNo, timeShiftVsRef, timeSeriesDf, spikes_h5_DF, fs, enablePrint, trajectoryOnly, Id_str, use_hpfAvg, process_hpf, processWindow, removeStimuliInterference=False, stimuliElectrodeNo=-1):
    '''
    #########################
    fig, axs = plt.subplots(1)
    bands = (0, 1000, 1500, 4000, 5000, fs/2,)
    desired = (0, 0, 1, 1,0,0)
    numtaps = 25#13
    fir_firls = signal.firls(numtaps, bands, desired, fs=fs)
    sos = signal.tf2sos(b=fir_firls, a=1)
    hs = list()
    ax = axs
    freq, response = signal.freqz(fir_firls)
    hs.append(ax.semilogy(0.5*fs*freq/np.pi, np.abs(response))[0])
    for band, gains in zip(zip(bands[::2], bands[1::2]),
                           zip(desired[::2], desired[1::2])):
        ax.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)
    ax.legend(hs, 'firls', loc='lower center', frameon=False)
    ax.set_xlabel('Frequency (Hz)')
    ax.grid(True)
    ax.set(title='High-pass %d-%d Hz' % bands[2:4], ylabel='Magnitude')
    
    fig.tight_layout()
    plt.show()
    ########################
    '''
    printNotOnlySavedFigures = False
    axx,ab = 16/3,9/3
    
    spikeRefChannelNo = mappingDF[mappingDF['electrode'] == spikeRefElectrodeNo]['channel'].to_numpy()[0].astype(int)
    timesOfSpikesInRefChannel = spikes_h5_DF[spikes_h5_DF['channel'] == spikeRefChannelNo].loc[:, ['time', 'amplitude']]

    processedElectrodeChannelNo = mappingDF[mappingDF['electrode'] == processedElectrodeNo]['channel'].to_numpy()[0].astype(int)

    if removeStimuliInterference:
        stimuliElcChannelNo = mappingDF[mappingDF['electrode'] == stimuliElectrodeNo]['channel'].to_numpy()[0].astype(int)
        timesOfSpikesInStimuliChannel = spikes_h5_DF[spikes_h5_DF['channel'] == stimuliElcChannelNo].loc[:, ['time', 'amplitude']]

    if not(timeSeriesDf is None):
        startOfRecordingsIndices = np.concatenate((np.zeros(1),np.where(timeSeriesDf['time'].diff() > 1/fs + 1e-6)[0])).astype(int)
        stopOfRecordingsIndices = np.concatenate((startOfRecordingsIndices[1:], np.array([timeSeriesDf.shape[0]]))).astype(int)
    


    
    if False and not(timeSeriesDf is None):
        fig, axs = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*10,ab*2*1), sharex=True, sharey=False)
        ax = axs
        for i,startIdx,stopIdx in zip(np.arange(startOfRecordingsIndices.shape[0]), startOfRecordingsIndices, stopOfRecordingsIndices):
            ax.plot(timeSeriesDf.loc[startIdx:stopIdx-1, 'time'], timeSeriesDf.loc[startIdx:stopIdx-1, 'e '+ str(spikeRefElectrodeNo)], 'k', label='raw')
            ax.plot(timeSeriesDf.loc[startIdx:stopIdx-1, 'time'], timeSeriesDf.loc[startIdx:stopIdx-1, 'e '+ str(spikeRefElectrodeNo) + ' hpf'], 'g', label='hpf')
        ax.grid()
        ax.set_xlabel('sec')
        #ax.set_title('Soma')
        plt.show()
    
    if use_hpfAvg:    
        path2mySpikeFile = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(spikeRefElectrodeNo) + '_mySpikes_hpf_DF.csv'
    else:
        path2mySpikeFile = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(spikeRefElectrodeNo) + '_mySpikes_DF.csv'
        
    if mySpikeDf is None:
        if os.path.isfile(path2mySpikeFile):
            mySpikeDf = pd.read_csv(path2mySpikeFile)
        else:
            
            selfSpikeEstimation = False
            firstSpikeFound = False
            for i,startIdx,stopIdx in zip(np.arange(startOfRecordingsIndices.shape[0]), startOfRecordingsIndices, stopOfRecordingsIndices):
                print(f'processElectrode: starting mySpikeDf {i} out of {startOfRecordingsIndices.shape[0]}')
                if i==0:
                    previousTrigTime, lastTrigTime = np.nan, np.nan
                else:
                    previousTrigTime = lastTrigTime
                
                
                singleTrigStartTime = timeSeriesDf.loc[startIdx,'time']
                singleTrigStopTime = timeSeriesDf.loc[stopIdx-1,'time']
                
                croppedTimesOfSpikesInRefChannel = timesOfSpikesInRefChannel[np.logical_and(timesOfSpikesInRefChannel['time'] >= singleTrigStartTime, timesOfSpikesInRefChannel['time'] <= singleTrigStopTime)]
                
                if croppedTimesOfSpikesInRefChannel.shape[0] > 0:
                    # spike was recorded in this channel
                    singleTrig = timeSeriesDf.loc[startIdx:stopIdx-1,'e '+ str(spikeRefElectrodeNo)].to_numpy()
                    singleTrigIdx = timeSeriesDf.loc[startIdx,'trigIdx']
                    
                    df = spikeFeaturesExtract(singleTrig, singleTrigStartTime, previousTrigTime, fs, singleTrigIdx, croppedTimesOfSpikesInRefChannel, selfSpikeEstimation)
                    if df.shape[0] > 0:
                        lastTrigTime = df['time'].to_numpy()[-1]
                    if not firstSpikeFound:
                        mySpikeDf = df
                        firstSpikeFound = True
                    else:
                        mySpikeDf = pd.concat([mySpikeDf, df], ignore_index=True)
            mySpikeDf.to_csv(path2mySpikeFile)
    
    
    # build a table with 1.2msec before the spike and 1.8msec after
    preSpikeTime = 1.0e-3 # sec
    if trajectoryOnly:
        postSpikeTime = processWindow#30e-3 # sec
    else:
        postSpikeTime = processWindow#30e-3#1.2e-3 # sec
    preSpikeSamples = int(np.round(preSpikeTime*fs))
    postSpikeSamples = int(np.round(postSpikeTime*fs))
    if np.mod(preSpikeSamples+postSpikeSamples, 2) == 0:
        postSpikeSamples = postSpikeSamples + 1
    N = preSpikeSamples+postSpikeSamples
    
    
    path2my_mySpikeDf_withSamples = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithSamples_DF.csv'
        
    if os.path.isfile(path2my_mySpikeDf_withSamples):
        mySpikeDf_withSamples = pd.read_csv(path2my_mySpikeDf_withSamples)
    else:
        print('processElectrode: starting mySpikeDf_withSamples')
        spikesTimeSeries = np.zeros((mySpikeDf.shape[0], 2*N))
        spikesTimeSeriesStartTimes = np.zeros((mySpikeDf.shape[0]))
        
        
        #duplicatedSpikeTimes = np.broadcast_to(mySpikeDf['time'] + timeShiftVsRef, (timeSeriesDf.shape[0], mySpikeDf['time'].shape[0]))
        #duplicatedTvec = np.broadcast_to(timeSeriesDf['time'], (mySpikeDf.shape[0], timeSeriesDf['time'].shape[0])).transpose()
        #peakIdxInTimesSeriesDf = np.abs(duplicatedSpikeTimes - duplicatedTvec).argmin(axis=0)
        
        coarse_timeSeriesDf = timeSeriesDf['time'][::int(fs)]

        for i in mySpikeDf.index:
            if np.mod(i, 1000) == 0:
                print(f'building mySpikeDf_withSamples {i} out of {mySpikeDf.index[-1]}')
            
            systemSpikeTime = mySpikeDf.loc[i, 'time']
            
            indexInCoarse = np.searchsorted(coarse_timeSeriesDf-systemSpikeTime - timeShiftVsRef, 0, 'left')
            startIndex, stopIndex = np.max([0, (indexInCoarse-1)*int(fs)]), np.min([timeSeriesDf.shape[0], int(fs)*(indexInCoarse+1)])
            peakIdxInTimesSeriesDf = startIndex + np.searchsorted(timeSeriesDf['time'][startIndex:stopIndex]-systemSpikeTime - timeShiftVsRef, 0, 'left')
            
            #peakIdxInTimesSeriesDf = previousPeakIdxInTimesSeriesDf + np.searchsorted(timeSeriesDf['time'][previousPeakIdxInTimesSeriesDf:]-systemSpikeTime - timeShiftVsRef, 0, 'left')
            previousPeakIdxInTimesSeriesDf = peakIdxInTimesSeriesDf
            peakStartIdxInTimeSeriesDf = peakIdxInTimesSeriesDf - preSpikeSamples
            peakPostIdxInTimeSeriesDf = peakIdxInTimesSeriesDf + postSpikeSamples
            #indices[peakStartIdxInTimeSeriesDf:peakPostIdxInTimeSeriesDf] = True
            spikesTimeSeries[i] = timeSeriesDf.loc[peakStartIdxInTimeSeriesDf:peakPostIdxInTimeSeriesDf-1,['e '+ str(processedElectrodeNo),'e '+ str(processedElectrodeNo) + ' hpf']].to_numpy().flatten(order='F')
            spikesTimeSeriesStartTimes[i] = timeSeriesDf.loc[peakStartIdxInTimeSeriesDf, 'time']
            
            times = timeSeriesDf.loc[peakStartIdxInTimeSeriesDf:peakPostIdxInTimeSeriesDf-1, 'time']#.to_numpy()
            splitIndices = np.where(np.diff(times)>1/fs+1e-6)[0]

            if removeStimuliInterference and not(stimuliElectrodeNo == processedElectrodeNo):
                timesOfStimuliSpikes = timesOfSpikesInStimuliChannel['time'][np.logical_and(timesOfSpikesInStimuliChannel['time'] <= times.iloc[-1], timesOfSpikesInStimuliChannel['time'] >= times.iloc[0])]
                removeStartTimes, removeStopTimes = timesOfStimuliSpikes - 1.8e-3, timesOfStimuliSpikes + 1.8e-3
                removeStartTimes[removeStartTimes < times.iloc[0]] = times.iloc[0]
                removeStopTimes[removeStopTimes > times.iloc[-1]] = times.iloc[-1]

                for ir in range(removeStartTimes.shape[0]):
                    removeStartTime, removeStopTime = removeStartTimes.iloc[ir], removeStopTimes.iloc[ir]
                    removeStartIdx = np.max([0, np.searchsorted(times - removeStartTime, 0, 'left')])
                    removeStopIdx = np.min([len(times), np.searchsorted(times - removeStopTime, 0, 'left')])
                    spikesTimeSeries[i, removeStartIdx:removeStopIdx] = np.nan
                    spikesTimeSeries[i, removeStartIdx + N:removeStopIdx + N] = np.nan

            #if i < mySpikeDf.shape[0]-1:
            #    nextSystemSpikeTime = mySpikeDf.loc[i+1, 'time']
            #else:
            #    nextSystemSpikeTime = np.inf
                
            
            #nextSystemSpikeTimeIdx = np.searchsorted(times-nextSystemSpikeTime, 0, 'right')
            #preSpikeSafety_nSamples = int(1e-3*fs)
            #if splitIndices.shape[0] > 0:
            #    lastIndex = np.min([splitIndices[0], nextSystemSpikeTimeIdx-preSpikeSafety_nSamples])
            #else:
            #    lastIndex = nextSystemSpikeTimeIdx-preSpikeSafety_nSamples
            
            #spikesTimeSeries[i, lastIndex:N] = np.nan
            #spikesTimeSeries[i, lastIndex+N:2*N] = np.nan
            
            if splitIndices.shape[0] > 0:
                lastIndex = splitIndices[0]
                spikesTimeSeries[i, lastIndex:N] = np.nan
                spikesTimeSeries[i, lastIndex+N:2*N] = np.nan
        #spikesTimeSeries = timeSeriesDf.loc[indices ,['e '+ str(processedElectrodeNo),'e '+ str(processedElectrodeNo) + ' hpf']]
        #spikesTimeSeries.loc[indices2zero[indices]] = 0

        spikesTimeSeriesDf = pd.DataFrame(data=np.concatenate((spikesTimeSeriesStartTimes[:, None], spikesTimeSeries), axis=1), columns = ['t(s=0)']+['s=' + str(s) for s in range(N)]+['hpf s=' + str(s) for s in range(N)])
        mySpikeDf_withSamples = pd.concat((mySpikeDf, spikesTimeSeriesDf), axis=1)
        '''
        controlTimeSeriesTimeShift = 2e-3 # sec
        controlTimeSeries = np.zeros((mySpikeDf_withSamples.shape[0], 2*N))
        for i in mySpikeDf_withSamples.index:
            #print(f'building control mySpikeDf_withSamples {i} out of {mySpikeDf.index[-1]}')
            peakIdxInTimesSeriesDf = np.argmin(np.abs(mySpikeDf_withSamples.loc[i, 'time'] + timeShiftVsRef + controlTimeSeriesTimeShift - timeSeriesDf['time']))
            peakStartIdxInTimeSeriesDf = peakIdxInTimesSeriesDf - preSpikeSamples
            peakPostIdxInTimeSeriesDf = peakIdxInTimesSeriesDf + postSpikeSamples
            controlTimeSeries[i] = np.concatenate((timeSeriesDf.loc[peakStartIdxInTimeSeriesDf:peakPostIdxInTimeSeriesDf-1, 'e '+ str(processedElectrodeNo)].to_numpy()[None,:], timeSeriesDf.loc[peakStartIdxInTimeSeriesDf:peakPostIdxInTimeSeriesDf-1, 'e '+ str(processedElectrodeNo) + ' hpf'].to_numpy()[None,:]), axis=1)
            
            times = timeSeriesDf.loc[peakStartIdxInTimeSeriesDf:peakPostIdxInTimeSeriesDf-1, 'time'].to_numpy()
            splitIndices = np.where(np.diff(times)>1/fs+1e-6)[0]
            if splitIndices.shape[0] > 0:
                lastIndex = splitIndices[0]
                controlTimeSeries[i, lastIndex:N] = 0
                controlTimeSeries[i, lastIndex+N:2*N] = 0
        controlTimeSeriesDf = pd.DataFrame(data = controlTimeSeries, columns = ['csc=' + str(s) for s in range(N)]+['hpf csc=' + str(s) for s in range(N)])
        mySpikeDf_withSamples = pd.concat((mySpikeDf_withSamples, controlTimeSeriesDf), axis=1)
        '''
        mySpikeDf_withSamples.to_csv(path2my_mySpikeDf_withSamples)
    
    
    
    tVec = np.arange(N)/fs # sec
    matshowAspect = 0.025/(mySpikeDf.shape[0]/4000)*(tVec[-1]/(8.5e-3))
    if enablePrint:
        for index, row in mySpikeDf_withSamples.iterrows():
            if index > 5:
                break
            if printNotOnlySavedFigures:
                fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*1*1), sharex=True, sharey=False)
                singleTimeSeries = row.loc['hpf s=0':f'hpf s={N-1}']
                plt.plot(tVec/1e-3, singleTimeSeries/1e-6, label=f'{index}' + '; hpf', linewidth=0.5)
                singleTimeSeries = row.loc['s=0':f's={N-1}']
                plt.plot(tVec/1e-3, singleTimeSeries/1e-6, label=f'{index}' + '; raw', linewidth=0.5)
                #plt.legend()
                #plt.axvline(x=preSpikeTime/1e-3, color='k', linestyle='dashed', linewidth=1)
                plt.ylabel('uv')
                plt.xlabel('msec')
                plt.legend()
                plt.title(Id_str + f': {index}')
                plt.grid()
                plt.show()


    
    full_pattern = mySpikeDf_withSamples.loc[:, 'hpf s=0':f'hpf s={N-1}'].mean()
    tVec = np.arange(N)/fs # sec
    if enablePrint:
        if printNotOnlySavedFigures:
            fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*1*1), sharex=True, sharey=False)
            plt.plot(tVec/1e-3, mySpikeDf_withSamples.loc[:,'hpf s=0':f'hpf s={N-1}'].to_numpy().transpose()/1e-6, label=index, linewidth=0.1)
            #plt.legend()
            plt.axvline(x=preSpikeTime/1e-3, color='k', linestyle='dashed', linewidth=1)
            plt.ylabel('uv')
            plt.xlabel('msec')
            plt.title(Id_str)
            plt.grid()
    
    if trajectoryOnly:
        if enablePrint:
            if printNotOnlySavedFigures:
                plt.plot(tVec/1e-3, full_pattern/1e-6, 'k', linewidth=2, label='average pattern')
                #plt.legend()
                plt.show()
        return full_pattern, mySpikeDf
    
    patternWidth = 1.4e-3 # sec
    nSamplesInPattern = int(np.round(patternWidth*fs))
    
    if not (np.abs(full_pattern)>0).any():
        return mySpikeDf, full_pattern
        
    patternCenterOfMass = int(np.round(np.dot(np.power(full_pattern[np.logical_not(full_pattern.isna())]-full_pattern.mean(), 2)/np.power(full_pattern[np.logical_not(full_pattern.isna())]-full_pattern.mean(), 2).sum(), np.arange(len(full_pattern[np.logical_not(full_pattern.isna())])))))
    initPatternStartIdx = int(np.round(patternCenterOfMass - nSamplesInPattern/2))
    initPatternStopIdx = initPatternStartIdx + nSamplesInPattern
    if full_pattern.isna().any() and initPatternStopIdx > np.where(full_pattern.isna())[0][0]:
        initPatternStopIdx = np.where(full_pattern.isna())[0][0]
        initPatternStartIdx = initPatternStopIdx - nSamplesInPattern
    pattern = full_pattern[initPatternStartIdx:initPatternStopIdx]
    pattern_tVec = tVec[initPatternStartIdx:initPatternStopIdx]
    patternScaled = (pattern-pattern.mean())/pattern.std()
    if enablePrint and printNotOnlySavedFigures:
        plt.plot(pattern_tVec/1e-3, pattern/1e-6, 'k', linewidth=2, label='average pattern')
    
    if use_hpfAvg:
        path2optimalPattern = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_optimalPattern_hpf_DF.csv'
    else:
        path2optimalPattern = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_optimalPattern_DF.csv'
    
    if os.path.isfile(path2optimalPattern):
        optimalPattern = pd.read_csv(path2optimalPattern)
        if 'Unnamed: 0' in optimalPattern.columns:
            optimalPattern.drop(columns=['Unnamed: 0'], inplace=True)
        optimalPattern = optimalPattern.to_numpy()[:,0]
    else:
        optimalPattern = patternOptimization(filename, analysisLibrary, nameOf_h5, configuration, processedElectrodeNo, mySpikeDf_withSamples, N, nSamplesInPattern, use_hpfAvg, False, Id_str, process_hpf)
        pd.Series(optimalPattern).to_csv(path2optimalPattern)
        
    if use_hpfAvg and enablePrint:
        fullPattern_hpf = mySpikeDf_withSamples.loc[:, 'hpf s=0':f'hpf s={N-1}'].mean()
        fullPattern_hpf = fullPattern_hpf - fullPattern_hpf.median()
        maxIdx = fullPattern_hpf.abs().argmax()
        startIdx = int(np.max([0,maxIdx-int(nSamplesInPattern/2)]))
        startIdx = int(np.min([len(fullPattern_hpf)-nSamplesInPattern,startIdx]))
        initPattern_hpf = fullPattern_hpf[startIdx:startIdx+nSamplesInPattern].to_numpy()
        
        if printNotOnlySavedFigures:
            plt.figure()
            pstartIdx = int(np.max([0,startIdx-int(3*nSamplesInPattern)]))
            pstopIdx = np.min([pstartIdx + int(7*nSamplesInPattern), len(fullPattern_hpf)])
            plt.plot(tVec[pstartIdx:pstopIdx]/1e-6, fullPattern_hpf[pstartIdx:pstopIdx], 'k', linewidth=0.5)
            plt.plot(tVec[startIdx:startIdx+nSamplesInPattern]/1e-6, initPattern_hpf, 'r', linewidth=1.0)
            plt.title(Id_str + ' hpf avg pattern')
            plt.xlabel('us')
            plt.grid()
            plt.show()
        
        hpf_ts_sub_meidan = mySpikeDf_withSamples.loc[:, 'hpf s=0':f'hpf s={N-1}'] - np.repeat(mySpikeDf_withSamples.loc[:, 'hpf s=0':f'hpf s={N-1}'].median(axis=1).to_numpy()[:,None], N, 1)
        noiseVar = hpf_ts_sub_meidan.loc[:, 'hpf s=100':f'hpf s={N-1}'].to_numpy().flatten().var()
        hpf_avg = hpf_ts_sub_meidan.loc[:, 'hpf s=0':f'hpf s={N-1}'].mean()
        avgNoiseVar = hpf_avg.to_numpy().flatten().var()
        print(f' noiseVar/avgNoiseVar decreased by {noiseVar/avgNoiseVar} when averaging {hpf_ts_sub_meidan.shape[0]} time-series')
        print(f'p2p in avg sig is {(hpf_avg.max()-hpf_avg.min())/1e-6}uv')
        
        ts_sub_meidan = mySpikeDf_withSamples.loc[:, 's=0':f's={N-1}'] - np.repeat(mySpikeDf_withSamples.loc[:, 's=0':f's={N-1}'].median(axis=1).to_numpy()[:,None], N, 1)
        noiseVar = ts_sub_meidan.loc[:, 's=100':f's={N-1}'].to_numpy().flatten().var()
        avg = ts_sub_meidan.loc[:, 's=0':f's={N-1}'].mean()
        avgNoiseVar = avg.to_numpy().flatten().var()
        print(f' noiseVar/avgNoiseVar decreased by {noiseVar/avgNoiseVar} when averaging {ts_sub_meidan.shape[0]} time-series')
        print(f'p2p in avg sig is {avg.max()-avg.min()}')
        '''
        plt.figure()
        pstartIdx = int(np.max([0,startIdx-int(3*nSamplesInPattern)]))
        pstopIdx = np.min([pstartIdx + int(7*nSamplesInPattern), len(fullPattern_hpf)])
        plt.plot(tVec[pstartIdx:pstopIdx]/1e-6, signal.sosfilt(sos, fullPattern_hpf[pstartIdx:pstopIdx]), 'k', linewidth=0.5)
        plt.plot(tVec[startIdx:startIdx+nSamplesInPattern]/1e-6, signal.sosfilt(sos, initPattern_hpf), 'r', linewidth=1.0)
        plt.title(Id_str + ' hpf avg pattern' + f' hpf@{bands[1]}')
        plt.xlabel('us')
        plt.grid()
        plt.show()
        '''
    
    optimalPattern_vs_pattern_dtoa = np.argmax(normalizedCorrelator(full_pattern[np.logical_not(full_pattern.isna())].to_numpy()[None,:], optimalPattern)[0])
    optimalPatternScaled = (optimalPattern-optimalPattern.mean())/optimalPattern.std()*pattern.std() + pattern.mean()
    if enablePrint:
        if printNotOnlySavedFigures:
            plt.plot(tVec[optimalPattern_vs_pattern_dtoa:optimalPattern_vs_pattern_dtoa+len(optimalPattern)]/1e-3, optimalPatternScaled/1e-6, color='red', linewidth=2, label='optimized pattern')
            #plt.legend()
            plt.show()
    
    if enablePrint:
        '''
        fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*1*1), sharex=True, sharey=False)
        if process_hpf:
            plt.plot(tVec/1e-3, mySpikeDf_withSamples.loc[:,'hpf csc=0':f'hpf csc={N-1}'].to_numpy().transpose()/1e-6, label=index, linewidth=0.1)
            #plt.legend()
            controlPattern = mySpikeDf_withSamples.loc[:, 'hpf csc=0':f'hpf csc={N-1}'].mean()
        else:
            plt.plot(tVec/1e-3, mySpikeDf_withSamples.loc[:,'csc=0':f'csc={N-1}'].to_numpy().transpose()/1e-6, label=index, linewidth=0.1)
            #plt.legend()
            controlPattern = mySpikeDf_withSamples.loc[:, 'csc=0':f'csc={N-1}'].mean()
        plt.plot(tVec/1e-3, controlPattern/1e-6, 'k', linewidth=2)
        plt.axvline(x=preSpikeTime/1e-3, color='k', linestyle='dashed', linewidth=1)
        plt.ylabel('uv')
        plt.xlabel('msec')
        plt.title(f'Control; ' + Id_str)
        plt.show()
        '''
        
        fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*1*1), sharex=True, sharey=False)
        plt.plot(tVec[0:0+len(optimalPattern)]/1e-3, optimalPatternScaled/1e-6, color='red', label='optimized pattern')
        plt.ylabel('uv')
        plt.xlabel('msec')
        plt.legend()
        plt.title(Id_str)
        plt.savefig(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_optimizedPattern.png', dpi=150)
        plt.close()
    
    
    
    
    
    if process_hpf:
        corrsOptimal, _, corrsOptimalPolyFitLocs = normalizedCorrelator(mySpikeDf_withSamples.loc[:,'hpf s=0':f'hpf s={N-1}'].to_numpy(), optimalPattern, True)
    else:
        corrsOptimal, _, corrsOptimalPolyFitLocs = normalizedCorrelator(mySpikeDf_withSamples.loc[:,'s=0':f's={N-1}'].to_numpy(), optimalPattern, True)
    corrs_tVec = np.arange(corrsOptimal.shape[1])/fs
    corrsOptimalPolyFitTimes = corrsOptimalPolyFitLocs[0]/fs
    corrsOptimalDf = pd.DataFrame(data=np.concatenate((mySpikeDf_withSamples['time'].to_numpy()[:,None], corrsOptimal), axis=1), columns=['time'] + [str(round(x/1e-6,1))+' us' for x in corrs_tVec.tolist()])
    
    if use_hpfAvg:
        path2corrsOptimalPattern = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPattern_hpf_DF.csv'
    else:
        path2corrsOptimalPattern = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPattern_DF.csv'
    
    corrsOptimalDf.to_csv(path2corrsOptimalPattern)
    
    
    
    peaksCorrsOptimal, locsCorrsOptimal = Corerlator_ANN(torch.zeros((1,5))).polyFitCorr(torch.from_numpy(corrsOptimal).unsqueeze(-1).type(torch.float))
    
    if enablePrint:
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False)
        ax.plot(corrs_tVec/1e-3, corrsOptimal.transpose(), linewidth=0.1)
        ax.plot(corrs_tVec/1e-3, pd.DataFrame(corrsOptimal).mean(axis=0), 'k', linewidth=2)
        ax.set_xlabel('msec')
        ax.set_title(f'corr (optimal) for electrode {processedElectrodeNo}')
        ax.grid()
        fig.tight_layout()
        plt.savefig(analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPattern.png', dpi=150)
        plt.close()
        if printNotOnlySavedFigures:
            
            
            plt.figure()
            plt.plot(corrs_tVec/1e-3, corrsOptimal.mean(axis=0), 'k', linewidth=1)
            plt.xlabel('msec')
            plt.title(f'corr (optimal) for electrode {processedElectrodeNo}')
            plt.grid()
            plt.show()
        
        
        markerThrQuantile = 1 - nSamplesInPattern/corrsOptimal.shape[1]/4
        markerThr = np.quantile(corrsOptimal[np.logical_not(np.isnan(corrsOptimal))], markerThrQuantile)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False)#, sharex=True)#, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
        x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
        n_yTicks, n_xTicks = 5, 10
        yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
        
        vmax = pd.Series(corrsOptimal.flatten()).abs().max()
        vmin = pd.Series(corrsOptimal.flatten()).abs().min()
        
        cax = ax.matshow(10*np.log10(np.abs(corrsOptimal)), aspect=matshowAspect)
        for p in range(corrsOptimal.shape[0]):
            if corrsOptimal[p][np.nanargmax(corrsOptimal[p])] >= markerThr:
                plt.plot(np.nanargmax(corrsOptimal[p]), p,'k+', markersize=1.0)
        ax.set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
        ax.set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3)[::xD])
        ax.set_xlabel('ms')
        ax.set_ylabel('sec')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title('optimal corr ' + Id_str)
        #fig.colorbar(cax)
        fig.tight_layout()
        #size = fig.get_size_inches()
        saveStr = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPatternMaxMarkers.png'
        print('saving ' + saveStr)
        plt.savefig(saveStr, dpi=150)
        plt.close()
        
        
        
        
        if use_hpfAvg:
            mySpikeDf_withSamplesAuthentic = mySpikeDf_withSamples.loc[:,'hpf s=0':f'hpf s={N-1}'].to_numpy()
        else:
            mySpikeDf_withSamplesAuthentic = mySpikeDf_withSamples.loc[:,'s=0':f's={N-1}'].to_numpy()
        mySpikeDf_withSamplesFlatten = mySpikeDf_withSamplesAuthentic.flatten()
        mySpikeDf_withSamplesFlattenPermuted = mySpikeDf_withSamplesFlatten[np.random.permutation(mySpikeDf_withSamplesFlatten.shape[0])]
        mySpikeDf_withSamplesPermuted = np.reshape(mySpikeDf_withSamplesFlattenPermuted, mySpikeDf_withSamples.loc[:,'hpf s=0':f'hpf s={N-1}'].shape)
        
        # surrounding electrodes:
        x, y = mappingDF[mappingDF['electrode'] == processedElectrodeNo]['x'].to_numpy()[0], mappingDF[mappingDF['electrode'] == processedElectrodeNo]['y'].to_numpy()[0]
        electrodeRes = 17.5
        surroundingElectrodes = mappingDF[np.logical_and(np.abs(mappingDF['x']-x)/electrodeRes <= 1, np.abs(mappingDF['y']-y)/electrodeRes <= 1)]['electrode'].tolist()
        surroundingElectrodes = [int(r) for r in surroundingElectrodes]
        
        rollingValues = np.power(2, np.arange(0,np.log2(corrsOptimal.shape[0]))[:-1]).astype('int').tolist()
        rollingValues = [int(rollingValue) for rollingValue in rollingValues if rollingValue <= 1024]
        medianReactionTime = None
        
        for rollingValue in rollingValues:
            
            mySpikeDf_withSamplesAuthentic_rolled = pd.DataFrame(mySpikeDf_withSamplesAuthentic).rolling(window=rollingValue, min_periods=1).mean().to_numpy()
            
            for surroundingElectrode in surroundingElectrodes:
                path2my_mySpikeDf_withSamples_surrounding = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(surroundingElectrode) + '_mySpikesWithSamples_DF.csv'
                mySpikeDf_withSamples_surrounding = pd.read_csv(path2my_mySpikeDf_withSamples_surrounding)
                if use_hpfAvg:
                    spatialElectrode = mySpikeDf_withSamples_surrounding.loc[:,'hpf s=0':f'hpf s={N-1}'].to_numpy()
                else:
                    spatialElectrode = mySpikeDf_withSamples_surrounding.loc[:,'s=0':f's={N-1}'].to_numpy()
                    
                spatialElectrode_rolled = pd.DataFrame(spatialElectrode).rolling(window=rollingValue, min_periods=1).mean().to_numpy()
                electrodeConvRolled, electrodeConvRolledPolyFitVals, electrodeConvRolledPolyFitLocs = normalizedCorrelator(mySpikeDf_withSamplesAuthentic_rolled.flatten()[None,:], spatialElectrode_rolled.flatten(), enablePolyFit=True, enableTypeConvFull=True)
                spatialDTOA = (electrodeConvRolledPolyFitLocs[0] - (electrodeConvRolled.shape[1]-1) + mySpikeDf_withSamplesAuthentic_rolled.flatten().shape[0]-1)/fs
                #spatialCorrTvec = np.arange(electrodeConvRolled.shape[1])
                #spatialCorrTvec = (spatialCorrTvec - spatialCorrTvec[-1] + mySpikeDf_withSamplesAuthentic_rolled.flatten().shape[0]-1)/fs
                
                electrodeVoltage = np.sqrt(np.nansum(np.power(mySpikeDf_withSamplesAuthentic_rolled.flatten()-np.nanmean(mySpikeDf_withSamplesAuthentic_rolled.flatten()), 2)))
                spatialElectrodeVoltage = np.sqrt(np.nansum(np.power(spatialElectrode_rolled.flatten() - np.nanmean(spatialElectrode_rolled.flatten()), 2)))
                spatialCorr = electrodeConvRolledPolyFitVals[0,0]/electrodeVoltage/spatialElectrodeVoltage
                
                x, y = mappingDF[mappingDF['electrode'] == processedElectrodeNo]['x'].to_numpy()[0], mappingDF[mappingDF['electrode'] == processedElectrodeNo]['y'].to_numpy()[0]
                xSurrounding, ySurrounding = mappingDF[mappingDF['electrode'] == surroundingElectrode]['x'].to_numpy()[0], mappingDF[mappingDF['electrode'] == surroundingElectrode]['y'].to_numpy()[0]
                xDiff, yDiff = (xSurrounding-x)/electrodeRes, (ySurrounding-y)/electrodeRes
                
                mySpikeDf.insert(loc=mySpikeDf.shape[1], column=f'spatialDTOA@MA {rollingValue} @e {processedElectrodeNo} with ({str(round(xDiff))},{str(round(yDiff))})', value=spatialDTOA*np.ones((mySpikeDf.shape[0],1)))
                mySpikeDf.insert(loc=mySpikeDf.shape[1], column=f'spatialCORR@MA {rollingValue} @e {processedElectrodeNo} with ({str(round(xDiff))},{str(round(yDiff))})', value=spatialCorr*np.ones((mySpikeDf.shape[0],1)))
                
                    
            corrsOptimalRolled, _, corrsOptimalRolledPolyFitLocs = normalizedCorrelator(mySpikeDf_withSamplesAuthentic_rolled, optimalPattern, True)
            corrsOptimalRolledPolyFitTimes = corrsOptimalRolledPolyFitLocs[0]/fs
            corrsOptimalRolledPermuted = normalizedCorrelator(pd.DataFrame(mySpikeDf_withSamplesPermuted).rolling(window=rollingValue, min_periods=1).mean().to_numpy(), optimalPattern, False)
            
            if not (np.abs(corrsOptimalRolled)>0).any():
                continue
            
            # the last valid one is taken:
            medianReactionTime = pd.Series(corrsOptimalRolledPolyFitTimes).median()
                
            markerThrQuantile = 0
            markerThr = np.quantile(corrsOptimalRolled[np.logical_not(np.isnan(corrsOptimalRolled))], markerThrQuantile)
            
            fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(8,3), sharey=True)#, sharex=True)#, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
            plt.suptitle('optimal corr ' + Id_str)
            vmin = np.min([pd.Series(corrsOptimalRolled.flatten()).abs().min(), pd.Series(corrsOptimalRolledPermuted.flatten()).abs().min()])
            vmax = np.max([pd.Series(corrsOptimalRolled.flatten()).abs().max(), pd.Series(corrsOptimalRolledPermuted.flatten()).abs().max()])
            x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
            n_yTicks, n_xTicks = 5, 5
            yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
            
            cax = ax[0].matshow(10*np.log10(np.abs(corrsOptimalRolled)), aspect=matshowAspect)
            
            j = 0
            for p in range(corrsOptimalRolled.shape[0]):
                ax[0].plot(np.nanargmax(corrsOptimalRolled[p]), p,'k+', markersize=1.0)
                if corrsOptimalRolled[p].max() < corrsOptimalRolledPermuted[p].max():
                    ax[1].plot(np.nanargmax(corrsOptimalRolledPermuted[p]), p,'k+', markersize=1.0)
                    j += 1
            ax[0].set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
            ax[0].set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3,1)[::xD])
            ax[0].set_xlabel('ms')
            ax[0].set_ylabel('sec')
            ax[0].xaxis.set_ticks_position('bottom')
            ax[0].set_title(f'window = {str(round(rollingValue))}')
            
            cax = ax[1].matshow(10*np.log10(np.abs(corrsOptimalRolledPermuted)), aspect=matshowAspect)
            
            ax[1].set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
            ax[1].set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3,1)[::xD])
            ax[1].set_xlabel('ms')
            ax[1].set_title(f'permuted; {str(round(j/corrsOptimalRolled.shape[0]*100))}% of maximas')
            #ax[1].set_ylabel('sec')
            ax[1].xaxis.set_ticks_position('bottom')
            
            fig.tight_layout()
            
            #size = fig.get_size_inches()
            saveStr = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_movingAvg_' + str(rollingValue) + '_corrsOptimalPatternMaxMarkers.png'
            print('saving ' + saveStr)
            plt.savefig(saveStr, dpi=150)
            plt.close()
            
            fig, ax = plt.subplots(nrows=3, ncols=2, constrained_layout=False, sharey=True)
            plt.suptitle('optimal corr ' + Id_str)
            ax[0,0].plot(corrs_tVec/1e-3, corrsOptimalRolled.transpose(), linewidth=0.1)
            ax[0,0].plot(corrs_tVec/1e-3, pd.DataFrame(corrsOptimalRolled).mean(axis=0), 'k', linewidth=2)
            ax[0,0].set_xlabel('msec')
            #ax.set_title(f'corr (optimal) for electrode {processedElectrodeNo}')
            ax[0,0].grid()
            ax[0,1].plot(corrs_tVec/1e-3, corrsOptimalRolledPermuted.transpose(), linewidth=0.1)
            ax[0,1].plot(corrs_tVec/1e-3, pd.DataFrame(corrsOptimalRolledPermuted).mean(axis=0), 'k', linewidth=2)
            ax[0,1].set_xlabel('msec')
            ax[0,1].set_title('permuted')
            ax[0,1].grid()
            plt.subplot(3,1,2)
            maxTimes = corrsOptimalRolledPolyFitTimes
            thr_maxTimes = 225e-6
            maxTimes[np.abs(maxTimes-pd.Series(maxTimes).median()) > thr_maxTimes] = np.nan
            plt.plot(mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0], maxTimes/1e-6, 'k', linewidth=0.5)
            plt.xlabel('sec; ' + r'$\sigma=$'+f'{str(round(pd.Series(maxTimes).std()/1e-6, 1))} us')
            plt.ylabel('us')
            plt.grid()            
            #plt.legend()
            plt.title(f'window = {str(round(rollingValue))}; ' + r'detections within $\pm$' + f'{str(round(thr_maxTimes/1e-6))} us')
            plt.subplot(3,1,3)
            maxTimes = corrsOptimalRolledPolyFitTimes
            maxTimes[np.abs(maxTimes-np.median(maxTimes)) > thr_maxTimes] = np.nan
            plt.plot(maxTimes/1e-6, 'k', linewidth=0.5)
            plt.xlabel('events')
            plt.ylabel('us')
            plt.grid()
            fig.tight_layout()
            saveStr = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_movingAvg_' + str(rollingValue) + '_corrsOptimalPattern.png'
            print('saving ' + saveStr)
            plt.savefig(saveStr, dpi=150)
            plt.close()
            
            mySpikeDf.insert(loc=mySpikeDf.shape[1], column=f'toa@MA {rollingValue} @e {processedElectrodeNo}', value=corrsOptimalRolledPolyFitTimes)
            mySpikeDf.insert(loc=mySpikeDf.shape[1], column=f'toa_std@MA {rollingValue} @e {processedElectrodeNo}', value=pd.Series(corrsOptimalRolledPolyFitTimes).std()*np.ones(mySpikeDf.shape[0]))
            
        # on spatial avg:
        
        
        
        spatialAvgAuthentic = np.zeros_like(mySpikeDf_withSamplesAuthentic)
        
        meanSpatialCorrelation = list()
        medianReactionTimeShift = 1.5e-3 # sec
        if not(medianReactionTime is None):
            firstSampleForSpatialCorrelation = int((medianReactionTime+medianReactionTimeShift)*fs)
        else:
            firstSampleForSpatialCorrelation = None
        
        enableSpatialFigures = False
        if enableSpatialFigures:
            if not(firstSampleForSpatialCorrelation is None):
                for surroundingElectrode in surroundingElectrodes:
                    path2my_mySpikeDf_withSamples_surrounding = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(surroundingElectrode) + '_mySpikesWithSamples_DF.csv'
                    mySpikeDf_withSamples_surrounding = pd.read_csv(path2my_mySpikeDf_withSamples_surrounding)
                    if use_hpfAvg:
                        spatialElectrode = mySpikeDf_withSamples_surrounding.loc[:,'hpf s=0':f'hpf s={N-1}'].to_numpy()
                    else:
                        spatialElectrode = mySpikeDf_withSamples_surrounding.loc[:,'s=0':f's={N-1}'].to_numpy()
                    
                    if not surroundingElectrode==processedElectrodeNo:
                        meanSpatialCorrelation.append(pd.Series(mySpikeDf_withSamplesAuthentic[:,firstSampleForSpatialCorrelation:].flatten()).corr(pd.Series(spatialElectrode[:,firstSampleForSpatialCorrelation:].flatten())))
                    
                    spatialAvgAuthentic = spatialAvgAuthentic + spatialElectrode
                    
                spatialAvgAuthentic = spatialAvgAuthentic/len(surroundingElectrodes)
                
                spatialAvgAuthenticFlatten = spatialAvgAuthentic.flatten()
                spatialAvgAuthenticFlattenPermuted = spatialAvgAuthenticFlatten[np.random.permutation(spatialAvgAuthenticFlatten.shape[0])]
                spatialAvgAuthenticPermuted = np.reshape(spatialAvgAuthenticFlattenPermuted, mySpikeDf_withSamples.loc[:,'hpf s=0':f'hpf s={N-1}'].shape)
                
                for rollingValue in rollingValues:
                    corrsOptimalRolled, _, corrsOptimalRolledPolyFitLocs = normalizedCorrelator(pd.DataFrame(spatialAvgAuthentic).rolling(window=rollingValue, min_periods=1).mean().to_numpy(), optimalPattern, True)
                    corrsOptimalRolledPolyFitTimes = corrsOptimalRolledPolyFitLocs[0]/fs
                    corrsOptimalRolledPermuted = normalizedCorrelator(pd.DataFrame(spatialAvgAuthenticPermuted).rolling(window=rollingValue, min_periods=1).mean().to_numpy(), optimalPattern, False)
                    
                    if not (np.abs(corrsOptimalRolled)>0).any():
                        continue
                    
                    markerThrQuantile = 0
                    markerThr = np.quantile(corrsOptimalRolled[np.logical_not(np.isnan(corrsOptimalRolled))], markerThrQuantile)
                    
                    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(8,3), sharey=True)#, sharex=True)#, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
                    plt.suptitle('optimal corr ' + Id_str)
                    vmin = np.min([pd.Series(corrsOptimalRolled.flatten()).abs().min(), pd.Series(corrsOptimalRolledPermuted.flatten()).abs().min()])
                    vmax = np.max([pd.Series(corrsOptimalRolled.flatten()).abs().max(), pd.Series(corrsOptimalRolledPermuted.flatten()).abs().max()])
                    x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
                    n_yTicks, n_xTicks = 5, 5
                    yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
                    
                    cax = ax[0].matshow(10*np.log10(np.abs(corrsOptimalRolled)), aspect=matshowAspect)
                        
                       
                    j = 0
                    for p in range(corrsOptimalRolled.shape[0]):
                        ax[0].plot(np.nanargmax(corrsOptimalRolled[p]), p,'k+', markersize=1.0)
                        if corrsOptimalRolled[p].max() < corrsOptimalRolledPermuted[p].max():
                            ax[1].plot(np.nanargmax(corrsOptimalRolledPermuted[p]), p,'k+', markersize=1.0)
                            j += 1
                    ax[0].set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
                    ax[0].set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3,1)[::xD])
                    ax[0].set_xlabel('ms')
                    ax[0].set_ylabel('sec')
                    ax[0].xaxis.set_ticks_position('bottom')
                    ax[0].set_title(f'window = {str(round(rollingValue))}; spatial corr = {str(round(np.asarray(meanSpatialCorrelation).mean(),1))}'+r'$\pm$'+f'{str(round(np.asarray(meanSpatialCorrelation).std(),1))}')
                    
                    cax = ax[1].matshow(10*np.log10(np.abs(corrsOptimalRolledPermuted)), aspect=matshowAspect)
                    
                    ax[1].set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
                    ax[1].set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3,1)[::xD])
                    ax[1].set_xlabel('ms')
                    ax[1].set_title(f'permuted; {str(round(j/corrsOptimalRolled.shape[0]*100))}% of maximas')
                    #ax[1].set_ylabel('sec')
                    ax[1].xaxis.set_ticks_position('bottom')
                    
                    fig.tight_layout()
                    
                    #size = fig.get_size_inches()
                    saveStr = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_movingAvg_' + str(rollingValue) + '_spatial_corrsOptimalPatternMaxMarkers.png'
                    print('saving ' + saveStr)
                    plt.savefig(saveStr, dpi=150)
                    plt.close()
                    
                    fig, ax = plt.subplots(nrows=3, ncols=2, constrained_layout=False, sharey=True)
                    plt.suptitle('optimal corr ' + Id_str)
                    ax[0,0].plot(corrs_tVec/1e-3, corrsOptimalRolled.transpose(), linewidth=0.1)
                    ax[0,0].plot(corrs_tVec/1e-3, pd.DataFrame(corrsOptimalRolled).mean(axis=0), 'k', linewidth=2)
                    ax[0,0].set_xlabel('msec')
                    #ax.set_title(f'corr (optimal) for electrode {processedElectrodeNo}')
                    ax[0,0].grid()
                    ax[0,1].plot(corrs_tVec/1e-3, corrsOptimalRolledPermuted.transpose(), linewidth=0.1)
                    ax[0,1].plot(corrs_tVec/1e-3, pd.DataFrame(corrsOptimalRolledPermuted).mean(axis=0), 'k', linewidth=2)
                    ax[0,1].set_xlabel('msec')
                    ax[0,1].set_title('permuted')
                    ax[0,1].grid()
                    plt.subplot(3,1,2)
                    maxTimes = corrsOptimalRolledPolyFitTimes
                    #thr_maxTimes = 200e-6
                    maxTimes[np.abs(maxTimes-np.median(maxTimes)) > thr_maxTimes] = np.nan
                    plt.plot(mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0], maxTimes/1e-6, 'k', linewidth=0.5)
                    plt.xlabel('sec')
                    plt.ylabel('us')
                    plt.grid()
                    plt.title(f'window = {str(round(rollingValue))}; ' + f'detections within {str(round(thr_maxTimes/1e-6))} us')
                    plt.subplot(3,1,3)
                    maxTimes = corrsOptimalRolledPolyFitTimes
                    #thr_maxTimes = 200e-6
                    maxTimes[np.abs(maxTimes-np.median(maxTimes)) > thr_maxTimes] = np.nan
                    plt.plot(maxTimes/1e-6, 'k', linewidth=0.5)
                    plt.xlabel('events')
                    plt.ylabel('us')
                    plt.grid()
                    fig.tight_layout()
                    saveStr = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_movingAvg_' + str(rollingValue) + '_spatial_corrsOptimalPattern.png'
                    print('saving ' + saveStr)
                    plt.savefig(saveStr, dpi=150)
                    plt.close()

        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False)#, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
        x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
        n_yTicks, n_xTicks = 5, 10
        yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
        
        vmax = pd.Series(corrsOptimal.flatten()).abs().max()
        vmin = pd.Series(corrsOptimal.flatten()).abs().min()
        
        cax = ax.matshow(10*np.log10(np.abs(corrsOptimal)), aspect=matshowAspect)
        ax.set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
        ax.set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3)[::xD])
        ax.set_xlabel('ms')
        ax.set_ylabel('sec')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title('optimal corr ' + Id_str)
        #fig.colorbar(cax)
        fig.tight_layout()
        saveStr = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPatternHeatmap.png'
        print('saving ' + saveStr)
        plt.savefig(saveStr, dpi=150)
        plt.close()
        
        
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False)#, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
        qmin, qmax = 5/100, 95/100
        x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
        n_yTicks, n_xTicks = 5, 10
        yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
        #cax = ax.matshow(np.zeros_like(corrsOptimal), aspect=matshowAspect, vmin=np.quantile(corrsOptimal, qmin), vmax=np.quantile(corrsOptimal, qmax))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
        cax = ax.matshow(np.zeros_like(corrsOptimal), aspect=matshowAspect, vmin=np.quantile(corrsOptimal, qmin), vmax=np.quantile(corrsOptimal, qmax))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
        for p in range(corrsOptimal.shape[0]):
            plt.plot(np.nanargmax(np.abs(corrsOptimal[p])), p,'k+', markersize=0.3)
        ax.set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
        ax.set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3)[::xD])
        ax.set_xlabel('ms')
        ax.set_ylabel('sec')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title('extreme optimal corr ' + Id_str)
        #fig.colorbar(cax)
        fig.tight_layout()
        saveStr = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_corrsOptimalPatternHeatmap_extreame.png'
        print('saving ' + saveStr)
        plt.savefig(saveStr, dpi=150)
        plt.close()
    
        
        if printNotOnlySavedFigures:
            fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False)#, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
            qmin, qmax = 5/100, 95/100
            x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
            n_yTicks, n_xTicks = 5, 10
            yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
            #cax = ax.matshow(corrsOptimal, aspect=matshowAspect, vmin=np.quantile(corrsOptimal, qmin), vmax=np.quantile(corrsOptimal, qmax))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
            cax = ax.matshow(corrsOptimal, aspect=matshowAspect, vmin=np.quantile(corrsOptimal, qmin), vmax=np.quantile(corrsOptimal, qmax))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
            ax.set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
            ax.set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3)[::xD])
            ax.set_xlabel('ms')
            ax.set_ylabel('sec')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title('optimal corr ' + Id_str)
            fig.colorbar(cax)
            fig.tight_layout()
            plt.show()
            
            
            fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
            qmin, qmax = 5/100, 95/100
            x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
            n_yTicks, n_xTicks = 5, 10
            yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
            #cax = ax.matshow(np.zeros_like(corrsOptimal), aspect=matshowAspect, vmin=np.quantile(corrsOptimal, qmin), vmax=np.quantile(corrsOptimal, qmax))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
            cax = ax.matshow(np.zeros_like(corrsOptimal), vmin=np.quantile(corrsOptimal, qmin), vmax=np.quantile(corrsOptimal, qmax))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
            for p in range(corrsOptimal.shape[0]):
                plt.plot(locsCorrsOptimal[p], p,'k+', markersize=0.3)
                
            ax.set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
            ax.set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3)[::xD])
            ax.set_xlabel('ms')
            ax.set_ylabel('sec')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title('max optimal corr ' + Id_str)
            fig.colorbar(cax)
            fig.tight_layout()
            plt.show()
        
        
        
            fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
            qmin, qmax = 5/100, 95/100
            x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
            n_yTicks, n_xTicks = 5, 10
            yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
            #cax = ax.matshow(np.zeros_like(corrsOptimal), aspect=matshowAspect, vmin=np.quantile(corrsOptimal, qmin), vmax=np.quantile(corrsOptimal, qmax))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
            cax = ax.matshow(np.zeros_like(corrsOptimal), vmin=np.quantile(corrsOptimal, qmin), vmax=np.quantile(corrsOptimal, qmax))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
            for p in range(corrsOptimal.shape[0]):
                plt.plot(corrsOptimal[p].argmin(), p,'k+', markersize=0.3)
            ax.set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
            ax.set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3)[::xD])
            ax.set_xlabel('ms')
            ax.set_ylabel('sec')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title('min optimal corr ' + Id_str)
            fig.colorbar(cax)
            fig.tight_layout()
            plt.show()
        
            fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False)#, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
            x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
            n_yTicks, n_xTicks = 5, 10
            yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
            #cax = ax.matshow(corrsOptimal, aspect=matshowAspect, vmin=np.quantile(corrsOptimal, 0), vmax=np.quantile(corrsOptimal, 1))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
            cax = ax.matshow(corrsOptimal, aspect=matshowAspect, vmin=np.quantile(corrsOptimal, 0), vmax=np.quantile(corrsOptimal, 1))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
            ax.set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
            ax.set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3)[::xD])
            ax.set_xlabel('ms')
            ax.set_ylabel('sec')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title('optimal corr ' + Id_str)
            fig.colorbar(cax)
            fig.tight_layout()
            plt.show()
            
            fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
            x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
            n_yTicks, n_xTicks = 5, 10
            yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
            #cax = ax.matshow(np.abs(corrsOptimal), aspect=matshowAspect, norm=colors.PowerNorm(gamma=4, vmin=np.quantile(np.abs(corrsOptimal), 0), vmax=np.quantile(np.abs(corrsOptimal), 1)))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
            cax = ax.matshow(np.abs(corrsOptimal), norm=colors.PowerNorm(gamma=4, vmin=np.quantile(np.abs(corrsOptimal), 0), vmax=np.quantile(np.abs(corrsOptimal), 1)))#, extent=[0,x_axis_duration/1e-3,y_axis_duration,0])
            ax.set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
            ax.set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3)[::xD])
            ax.set_xlabel('ms')
            ax.set_ylabel('sec')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title('optimal corr ' + Id_str)
            fig.colorbar(cax)
            fig.tight_layout()
            plt.show()
        
        
            fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False)#, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
            x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
            n_yTicks, n_xTicks = 5, 10
            yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
            
            cax = ax.matshow(10*np.log10(np.abs(corrsOptimal)), aspect=matshowAspect)
            ax.set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
            ax.set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3)[::xD])
            ax.set_xlabel('ms')
            ax.set_ylabel('sec')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title('optimal corr ' + Id_str)
            fig.colorbar(cax)
            fig.tight_layout()
            plt.show()
            
            fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*2*1,ab*2*1), sharex=True, sharey=False)
            x_axis_duration, y_axis_duration = corrs_tVec[-1]-corrs_tVec[0], mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[-1],'time']-mySpikeDf_withSamples.loc[mySpikeDf_withSamples.index[0],'time']
            n_yTicks, n_xTicks = 5, 10
            yD, xD = int(mySpikeDf_withSamples.index.shape[0]/n_yTicks), int(corrs_tVec.shape[0]/n_xTicks)
            
            cax = ax.matshow(10*np.log10(np.abs(corrsOptimal)))
            ax.set_yticks(np.arange(mySpikeDf_withSamples.index.shape[0])[::yD], np.round((mySpikeDf_withSamples['time']-mySpikeDf_withSamples['time'][0])[::yD]))
            ax.set_xticks(np.arange(corrs_tVec.shape[0])[::xD], np.round((corrs_tVec-corrs_tVec[0])/1e-3)[::xD])
            ax.set_xlabel('ms')
            ax.set_ylabel('sec')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title('optimal corr ' + Id_str)
            fig.colorbar(cax)
            fig.tight_layout()
            plt.show()
        
        
            fig, axs = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*6,ab*2))
            ax = axs#plt.subplot(fN,3,3+i*3)
            H, xedges, yedges = np.histogram2d(np.repeat(corrs_tVec, corrsOptimal.shape[0], 0)/1e-3, corrsOptimal.flatten(), bins=(corrs_tVec/1e-3, np.linspace(corrsOptimal.min(), corrsOptimal.max(), 100)), density=True)#, density=True)
            # Histogram does not follow Cartesian convention (see Notes),
            # therefore transpose H for visualization purposes.
            H = H.T
            X, Y = np.meshgrid(xedges, yedges)
            im = ax.pcolormesh(X, Y, H, cmap='rainbow')    
        
            #ax.set_xlim([logxbMin, logxbMax])
            #ax.set_ylim([alpha_T_min, alpha_T_max])
            ax.set_xlabel(r'ms',fontsize=16)
            #ax.set_ylabel(r'$T\alpha$',fontsize=16)
            #ax.set_title(r'$P();$' + r' $\rho(\log (x_b),T\alpha)=$' + f'{str(round(corr, 3))}' ,fontsize=16)
            plt.colorbar(im)
            ax.grid()
            plt.show()
    
    
    
    '''
    if process_hpf:
        controlCorrsOptimal = normalizedCorrelator(mySpikeDf_withSamples.loc[:,'hpf csc=0':f'hpf csc={N-1}'].to_numpy(), optimalPattern)
    else:
        controlCorrsOptimal = normalizedCorrelator(mySpikeDf_withSamples.loc[:,'csc=0':f'csc={N-1}'].to_numpy(), optimalPattern)
    corrs_tVec = np.arange(controlCorrsOptimal.shape[1])/fs
    
    if enablePrint:
        if printNotOnlySavedFigures:
            plt.figure()
            plt.plot(corrs_tVec/1e-3, controlCorrsOptimal.transpose(), linewidth=0.1)
            plt.plot(corrs_tVec/1e-3, controlCorrsOptimal.mean(axis=0), 'k', linewidth=2)
            plt.xlabel('msec')
            plt.title(f'Control; corr (optimal) for electrode {processedElectrodeNo}')
            plt.grid()
            plt.show()
    '''
    
    
    
    
    '''
    SNR = estimateSNR(mySpikeDf_withSamples, locsCorrsOptimal, N, len(optimalPattern), Id_str, process_hpf, enablePrint)
    SNR_db = 10*np.log10(SNR)
    print(f'electrode {processedElectrodeNo} SNR = {SNR_db} db')
    measurementJitterVar = np.power(np.sqrt(calcMeasNoiseTOA_var(SNR, N, optimalPattern))/fs, 2) # sec
    mySpikeDf.insert(loc=mySpikeDf.shape[1], column=f'~SNR_std@e {processedElectrodeNo}', value=np.sqrt(measurementJitterVar))
    mySpikeDf.insert(loc=mySpikeDf.shape[1], column=f'~SNR_db@e {processedElectrodeNo}', value=SNR_db)
    '''
    if enablePrint:
        if printNotOnlySavedFigures:
            for i in range(5):
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(corrs_tVec/1e-3, corrsOptimal[i], linewidth=0.5, label='corr')
                plt.plot(locsCorrsOptimal[i]/fs/1e-3, peaksCorrsOptimal[i], '*')
                plt.grid()
                plt.legend()
                plt.title(f'sanity check {i} ' + Id_str)
                plt.subplot(2,1,2)
                if process_hpf:
                    plt.plot(tVec/1e-3, mySpikeDf_withSamples.loc[i,'hpf s=0':f'hpf s={N-1}'].to_numpy(), linewidth=0.5, label='sig')
                else:
                    plt.plot(tVec/1e-3, mySpikeDf_withSamples.loc[i,'s=0':f's={N-1}'].to_numpy(), linewidth=0.5, label='sig')
                plt.xlabel('ms')
                plt.grid()
                plt.legend()
                plt.show()
                '''
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(corrs_tVec/1e-3, corrsOptimal[i], linewidth=0.5, label='corr')
                plt.plot(locsCorrsOptimal[i]/fs/1e-3, peaksCorrsOptimal[i], '*')
                plt.grid()
                plt.legend()
                plt.title(f'sanity check {i} ' + Id_str)
                plt.subplot(2,1,2)
                if process_hpf:
                    plt.plot(tVec[23:]/1e-3, signal.sosfilt(sos, mySpikeDf_withSamples.loc[i,'hpf s=0':f'hpf s={N-1}'].to_numpy())[23:], linewidth=0.5, label='sig bandpass')
                else:
                    plt.plot(tVec[23:]/1e-3, signal.sosfilt(sos, mySpikeDf_withSamples.loc[i,'s=0':f's={N-1}'].to_numpy())[23:], linewidth=0.5, label='sig bandpass')
                plt.xlabel('ms')
                plt.grid()
                plt.legend()
                plt.show()
                '''
    optimalAbsoluteTimeOfSpike = mySpikeDf_withSamples['t(s=0)'].to_numpy()+locsCorrsOptimal.detach().cpu().numpy()/fs
    #peaksControlCorrsOptimal, locsControlCorrsOptimal = Corerlator_ANN(torch.zeros((1,5))).polyFitCorr(torch.from_numpy(controlCorrsOptimal).unsqueeze(-1).type(torch.float))
    #optimalAbsoluteTimeOfSpikeControl = mySpikeDf_withSamples['t(s=0)'].to_numpy()+locsControlCorrsOptimal.detach().cpu().numpy()/fs
    if processedElectrodeNo == spikeRefElectrodeNo:
        optimalAbsoluteTimeOfSpike = optimalAbsoluteTimeOfSpike - (optimalAbsoluteTimeOfSpike-mySpikeDf['time']).median()
        #optimalAbsoluteTimeOfSpikeControl = optimalAbsoluteTimeOfSpikeControl - (optimalAbsoluteTimeOfSpike-mySpikeDf['time']).median()
    mySpikeDf.insert(loc=mySpikeDf.shape[1], column=f'~toas@e {processedElectrodeNo}', value=optimalAbsoluteTimeOfSpike)
    mySpikeDf.insert(loc=mySpikeDf.shape[1], column=f'~peaks@e {processedElectrodeNo}', value=peaksCorrsOptimal)
    
    referenceSpikeTimes = mySpikeDf[f'~toas@e {spikeRefElectrodeNo}'].to_numpy()
    deltaTimeFromRef = optimalAbsoluteTimeOfSpike - referenceSpikeTimes
    #deltaTimeFromRefControl = optimalAbsoluteTimeOfSpikeControl - referenceSpikeTimes
    
    mySpikeDf.insert(loc=mySpikeDf.shape[1], column=f'~dtoas@e {processedElectrodeNo}', value=deltaTimeFromRef)
    
    
    
    #controlCorrsOptimal_histCDF = plt.hist(controlCorrsOptimal.flatten(),bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=True,label=f'Control; Corr with optimal pattern over {str(round(N/fs/1e-3, 2))}ms CDF')
    #controlCorrsOptimal_histCDF_binsLeftEdges = controlCorrsOptimal_histCDF[1][:-1]
    #controlCorrsOptimal_histCDF_binsRightEdges = controlCorrsOptimal_histCDF[1][1:]
    #controlCorrsOptimal_histCDF_binsCenters = 0.5*(controlCorrsOptimal_histCDF_binsLeftEdges + controlCorrsOptimal_histCDF_binsRightEdges)
    #controlCorrsOptimal_histCDF_binsValues = controlCorrsOptimal_histCDF[0]
    '''
    # calculate false alarm probability per spike:
    faProb = np.zeros(len(peaksCorrsOptimal))
    for i, peaksCorrOptimal in enumerate(peaksCorrsOptimal.detach().cpu().numpy().tolist()):      
        if np.logical_or(peaksCorrOptimal <= controlCorrsOptimal_histCDF_binsLeftEdges[0], peaksCorrOptimal >= controlCorrsOptimal_histCDF_binsRightEdges[-1]):
            faProb[i] = 0
        else:
            b = np.where(np.logical_and(peaksCorrOptimal >= controlCorrsOptimal_histCDF_binsLeftEdges, peaksCorrOptimal < controlCorrsOptimal_histCDF_binsRightEdges))[0][0]
            faProb[i] = 1 - controlCorrsOptimal_histCDF_binsValues[b]
    mySpikeDf.insert(loc=mySpikeDf.shape[1], column=f'~Pfa@e {processedElectrodeNo}', value=faProb)
    '''
    meanDelay = deltaTimeFromRef.mean() # sec
    medianDelay = np.median(deltaTimeFromRef) # sec
    deltaTimeFromRefMin, deltaTimeFromRefMax = np.quantile(deltaTimeFromRef, 10/100), np.quantile(deltaTimeFromRef, 90/100)
    deltaTimeFromRefCropped = deltaTimeFromRef[np.logical_and(deltaTimeFromRef > deltaTimeFromRefMin, deltaTimeFromRef < deltaTimeFromRefMax)]
    
    if enablePrint:
        if printNotOnlySavedFigures:
            fig = plt.subplots(nrows=3, ncols=2, constrained_layout=False, figsize=(axx*1*4,ab*2*3), sharex=False, sharey=False)
            plt.subplot(3,3,1)
            plt.hist(peaksCorrsOptimal,bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=True,label=f'maxCorr with optimal pattern over {str(round(N/fs/1e-3, 2))}ms CDF')
            plt.legend()    
            #plt.title(Id_str)
            plt.grid()
            plt.xlabel('corr')
            plt.tight_layout()
            plt.subplot(3,3,2)
            plt.hist(peaksCorrsOptimal,bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=False,label=f'maxCorr with optimal pattern over {str(round(N/fs/1e-3, 2))}ms')
            #plt.hist(controlCorrsOptimal.flatten(),bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=False,label=f'Control; Corr with optimal pattern over {str(round(N/fs/1e-3, 2))}ms')
            plt.legend()    
            plt.title(Id_str)
            plt.grid()
            plt.xlabel('corr')
            plt.tight_layout()
            
            plt.subplot(6,3,3)
            plt.hist(mySpikeDf[f'~Pfa@e {processedElectrodeNo}'],bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=False, label='FA Prob PDF')
            plt.legend()
            #plt.xlabel('FA Prob')
            plt.grid()
            plt.tight_layout()
            
            plt.subplot(6,3,6)
            plt.hist(mySpikeDf[f'~Pfa@e {processedElectrodeNo}'],bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=True, label='FA Prob CDF')
            plt.legend()
            #plt.xlabel('FA Prob')
            plt.grid()
            plt.tight_layout()
        
            plt.subplot(3,2,3)
            
            plt.hist(deltaTimeFromRef/1e-6 - medianDelay/1e-6,bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=True,label=f'maxCorr with pattern over {str(round(N/fs/1e-3, 2))}ms')
            #plt.hist(deltaTimeFromRefControl/1e-6 - medianDelay/1e-6,bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=True,label=f'Control; maxCorr with pattern over {str(round(N/fs/1e-3, 2))}ms')
            plt.legend()    
            #plt.title(Id_str)
            plt.grid()
            plt.xlabel(r'$\Delta t$ from e ' + f'{spikeRefElectrodeNo} - median(={str(round(medianDelay/1e-6, 2))}) [us]')
            plt.tight_layout()
            plt.subplot(3,2,4)
            plt.hist(deltaTimeFromRef/1e-6 - medianDelay/1e-6,bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=False,label=f'maxCorr with pattern over {str(round(N/fs/1e-3, 2))}ms; std={str(round(deltaTimeFromRef.std()/1e-6,1))}us')
            #plt.hist(deltaTimeFromRefControl/1e-6 - medianDelay/1e-6,bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=False,label=f'Control; maxCorr with pattern over {str(round(N/fs/1e-3, 2))}ms')
            
            plt.hist(deltaTimeFromRefCropped/1e-6 - medianDelay/1e-6,bins=50,density=True,log=False,histtype='step',linewidth=1,cumulative=False,label=f'(80% of) maxCorr with pattern over {str(round(N/fs/1e-3, 2))}ms; std={str(round(deltaTimeFromRefCropped.std()/1e-6,1))}us')
            
            plt.hist(np.sqrt(measurementJitterVar)*np.random.randn(int(1e5))/1e-6,bins=100,linestyle='dashed', density=True,log=False,histtype='step',linewidth=1,cumulative=False,label=f'SNR ({str(round(SNR_db, 3))}db) jitter; TOAstd={str(round(np.sqrt(measurementJitterVar)/1e-6, 1))}us')
            
            plt.legend()    
            #plt.title(Id_str)
            plt.grid()
            plt.xlabel(r'$\Delta t$ from e ' + f'{spikeRefElectrodeNo} - median(={str(round(medianDelay/1e-6, 2))}) [us]')
            plt.tight_layout()
            plt.subplot(3,1,3)
            plt.scatter(x=deltaTimeFromRef/1e-6 - medianDelay/1e-6, y=peaksCorrsOptimal, s=1, label=f'maxCorr with optimal pattern over {str(round(N/fs/1e-3, 2))}ms')
            #plt.scatter(x=deltaTimeFromRefControl/1e-6 - medianDelay/1e-6, y=peaksControlCorrsOptimal, s=1, label=f'Control; maxCorr with optimal pattern over {str(round(N/fs/1e-3, 2))}ms')
            plt.xlabel(r'$\Delta t$ from e ' + f'{spikeRefElectrodeNo} - median(={str(round(medianDelay/1e-6, 2))}) [us]')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
            ########    TODO:  add process_hpf
            '''
            firstSpike = mySpikeDf_withSamples.loc[0, 's=0':f's={N-1}'].to_numpy()
            firstSpike - firstSpike.mean()
            fVec = fftshift(fftfreq(len(firstSpike), 1/fs))
            fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*1*1), sharex=True, sharey=False)
            fftRes = fftshift(fft(firstSpike))
            fftResPattern = fftshift(fft(pattern.to_numpy() - pattern.to_numpy().mean()))
            fVec_pattern = fftshift(fftfreq(len(pattern), 1/fs))
            plt.plot(fVec, 20*np.log10(np.abs(fftRes)) - (20*np.log10(np.abs(fftRes))).max(), label='first spike')
            plt.plot(fVec_pattern, 20*np.log10(np.abs(fftResPattern)) - (20*np.log10(np.abs(fftResPattern))).max(), label='pattern')
            plt.xlabel('hz')
            plt.legend()
            plt.title(Id_str)
            plt.show()
        
            
            allfftRes = np.zeros_like(mySpikeDf_withSamples.loc[:, 's=0':f's={N-1}'].to_numpy())
            for index, row in mySpikeDf_withSamples.iterrows():
                singleTimeSeries = row.loc['s=0':f's={N-1}']
                allfftRes[index] = fftshift(fft(singleTimeSeries.to_numpy()))
            fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*1*1), sharex=True, sharey=False)
            plt.plot(fVec, 20*np.log10(np.abs(allfftRes).sum(axis=0)), label='non-coherent sum')
            plt.plot(fVec, 20*np.log10(np.abs(allfftRes.sum(axis=0))), label='coherent sum')
            plt.xlabel('hz')
            plt.title(Id_str)
            plt.legend()
            plt.grid()
            plt.show()
            
            allMinusMeanfftRes = np.zeros_like(mySpikeDf_withSamples.loc[:, 's=0':f's={N-1}'].to_numpy())
            beforeSpikeIdx = np.argmin(np.abs(tVec-1e-3))
            for index, row in mySpikeDf_withSamples.iterrows():
                singleTimeSeries = row.loc['s=0':f's={N-1}']
                highAvg = singleTimeSeries[:beforeSpikeIdx].mean()
                singleTimeSeries = singleTimeSeries - highAvg
                allMinusMeanfftRes[index] = fftshift(fft(singleTimeSeries.to_numpy()))
            fig = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(axx*1*1,ab*1*1), sharex=True, sharey=False)
            plt.plot(fVec, 20*np.log10(np.abs(allMinusMeanfftRes).sum(axis=0)), label='subtract mean non-coherent sum')
            plt.plot(fVec, 20*np.log10(np.abs(allMinusMeanfftRes.sum(axis=0))), label='subtract mean coherent sum')
            plt.xlabel('hz')
            plt.title(Id_str)
            plt.legend()
            plt.grid()
            plt.show()
            '''
    
 
    if False:
        for i,startIdx,stopIdx in zip(np.arange(startOfRecordingsIndices.shape[0]), startOfRecordingsIndices, stopOfRecordingsIndices):
            
            singleTrigStartTime = timeSeriesDf.loc[startIdx,'time']
            singleTrigStopTime = timeSeriesDf.loc[stopIdx-1,'time']
            
            if (timesOfSpikesInRefChannel[np.logical_and(timesOfSpikesInRefChannel >= singleTrigStartTime, timesOfSpikesInRefChannel <= singleTrigStopTime)]).shape[0] > 0:
                # spike was recorded in this channel
                singleTrig = timeSeriesDf.loc[startIdx:stopIdx-1,'e '+ str(processedElectrodeNo)].to_numpy()
                
                singleTrigIdx = timeSeriesDf.loc[startIdx,'trigIdx']
                df = mySpikeDf[mySpikeDf['trigIdx'] == singleTrigIdx]
                printSpikeWithFeatures(singleTrig, singleTrigStartTime, df.reset_index(), fs, f'~toas@e {processedElectrodeNo}')
    
    path2my_mySpikeDf_withFeatures = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithFeatures_DF.csv'
    mySpikeDf.to_csv(path2my_mySpikeDf_withFeatures)
    
    path2my_mySpikeDf_withFeaturesSingleRow = analysisLibrary + nameOf_h5 + '_conf_' + configuration + '_electrode_' + str(processedElectrodeNo) + '_mySpikesWithFeaturesSingleRow_DF.csv'
    mySpikeDf.loc[0].to_csv(path2my_mySpikeDf_withFeaturesSingleRow)
    
    return mySpikeDf, optimalPattern



