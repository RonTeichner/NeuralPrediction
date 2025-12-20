import h5py
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm
import os
import matplotlib.pyplot as plt

EXPERIMENTS_DIR_PATH = "/raw_data"
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

BASE_PATH = "."
DATASET_DIR_PATH = os.path.abspath(os.path.join(BASE_PATH, "data/spikes_and_metrics_dfs"))
DATASET_FILE = "/dataset_somata.csv"

def read_h5(cache_dataset=True):
    dataset = pd.DataFrame()
    for experiment, experiment_path in tqdm(EXPERIMENTS_FILES.items()):
        hf = h5py.File(f"{EXPERIMENTS_DIR_PATH}/{experiment}/{experiment_path}", 'r', libver='latest', swmr=True)
        metrics = pd.read_excel(f"{EXPERIMENTS_DIR_PATH}/{experiment}/metrics_data.xlsx", sheet_name="Axon - Neuron Level", index_col=0)
        frame_nos = hf[f"{H5_DATA_HEADER}{H5_FRAME_NOS_KEY}"]
        mapping = hf[f"{H5_DATA_HEADER}{H5_MAPPING_KEY}"]
        fs = hf[f"{H5_DATA_HEADER}{H5_SETTINGS_SAMPLING}"][0]
        spikes = hf[f"{H5_DATA_HEADER}{H5_SPIKES_KEY}"]

        mapping_df = pd.DataFrame(np.array(mapping))

        spikes_df = pd.DataFrame(np.array(spikes))
        spikes_df["time"] = spikes_df["frameno"] / fs
        spikes_df = spikes_df[["time", "channel", "amplitude"]]
        spikes_df = spikes_df[spikes_df["channel"].isin(mapping_df["channel"].values)]
        spikes_df["electrode"] = mapping_df.set_index("channel").loc[spikes_df["channel"].values, :][
            "electrode"].values
        spikes_df["x"] = mapping_df.set_index("channel").loc[spikes_df["channel"].values, :]["x"].values
        spikes_df["y"] = mapping_df.set_index("channel").loc[spikes_df["channel"].values, :]["y"].values
        spikes_df["time"] = ((spikes_df["time"] * fs) - frame_nos[0]) / fs
        spikes_df = spikes_df[spikes_df["time"] >= 0]
        spikes_df["experiment"] = experiment

        metrics.to_csv(f"{DATASET_DIR_PATH}metrics_df__{experiment}.csv", index=False)
        spikes_df.to_csv(f"{DATASET_DIR_PATH}spikes_df__{experiment}.csv", index=False)

        somas_electrodes = [e for e in metrics["Electrode Number"].values if e in spikes_df["electrode"].values]
        somas_channels = spikes_df[spikes_df["electrode"].isin(somas_electrodes)]["channel"].unique()

        spikes_df = spikes_df[spikes_df["channel"].isin(somas_channels)]
        spikes_df = spikes_df.reset_index(drop=True)

        if len(dataset) == 0:
            dataset = spikes_df
        else:
            dataset = pd.concat([dataset, spikes_df])
    dataset = dataset.sort_values(by="time")
    dataset = dataset.reset_index(drop=True)
    if cache_dataset:
        dataset.to_csv(f"{DATASET_DIR_PATH}{DATASET_FILE}", index=False)
    return dataset


def show_single_voltage_trace_with_spike(experiment):
    dataset = pd.read_csv(f"{DATASET_DIR_PATH}{DATASET_FILE}")

    hf = h5py.File(f"{EXPERIMENTS_DIR_PATH}/{experiment}/{EXPERIMENTS_FILES[experiment]}", 'r', libver='latest', swmr=True)
    fs = hf[f"{H5_DATA_HEADER}{H5_SETTINGS_SAMPLING}"][0]

    channels = np.array(hf[f"{H5_DATA_HEADER}{H5_CHANNELS_KEY}"])
    raw = hf[f"{H5_DATA_HEADER}{H5_RAW_KEY}"]
    lsb = hf[f"{H5_DATA_HEADER}{H5_SETTINGS_LSB}"][0] / 1000  # Volt

    spikes_df = dataset[dataset["experiment"] == experiment]
    i = np.random.choice(spikes_df.shape[0])

    max_range = 200
    min_range = 20
    synced_frame = int(spikes_df.iloc[i]["time"] * fs)
    enum_channel = {c: ci for ci, c in enumerate(channels)}[int(spikes_df.iloc[i]["channel"])]
    amplitude = spikes_df.iloc[i]["amplitude"]

    voltage = raw[enum_channel, (synced_frame - max_range):(synced_frame + max_range + 1)].astype(float)
    voltage_volts = (voltage - np.median(voltage)) * lsb

    plt.figure()
    plt.plot(np.arange(-max_range, max_range + 1), voltage_volts)
    plt.plot([amplitude * lsb], "o", color="orange")
    plt.vlines(-min_range, voltage_volts.min(), voltage_volts.max(), color="black", linestyle="dashed")
    plt.vlines(min_range, voltage_volts.min(), voltage_volts.max(), color="black", linestyle="dashed")

    plt.legend(["Raw voltage", "Detected amplitude"])
    plt.xlabel("Frames around spike [20kHz bins]")
    plt.ylabel("Voltage [V]")
    plt.show()