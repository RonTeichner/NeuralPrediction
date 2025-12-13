import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
import os
from itertools import product

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import train_test_split

# Read:
BASE_PATH = ""
DATASET_DIR_PATH = os.path.join(BASE_PATH, "spikes_and_metrics_dfs")
DATASET_FILE = "/dataset_somata.csv"

class CacheReader:
    def __init__(self):
        pass

    @staticmethod
    def read(file_path=(DATASET_DIR_PATH + DATASET_FILE)):
        dataset = pd.read_csv(file_path)
        return dataset


# Prediction Scheme:
MIN_ISI = 0.005
PREPROCESS_START_OFFSET = 0
PREPROCESS_START_PERIOD = '2024-01-01 00:00:00'

def resolution_str(sample_time_resolution):
    return f"{sample_time_resolution * 1000000}U"


class PredictionScheme:
    def __init__(self, sample_time_resolution, min_isi=False):
        self.sample_time_resolution = sample_time_resolution
        self.min_isi = min_isi

    def process(self, dataset):
        if self.min_isi:
            dataset = dataset[dataset["value"] > MIN_ISI]

        metadata = dataset[["experiment", "channel", "electrode", "x", "y"]].drop_duplicates().reset_index()

        dataset["time_index"] = ((dataset["time"] - dataset["time"].min() - PREPROCESS_START_OFFSET) //
                                 self.sample_time_resolution)
        dataset = dataset[dataset["time_index"] >= 0]

        time_index_resolution = dataset["time_index"].values * pd.Timedelta(resolution_str(self.sample_time_resolution))
        time_index_shifted = pd.Timestamp(PREPROCESS_START_PERIOD) + pd.to_timedelta(time_index_resolution)
        period_index = pd.DatetimeIndex.to_period(time_index_shifted, resolution_str(self.sample_time_resolution))
        dataset["time_index"] = period_index

        dataset = self._process(dataset)

        complete_periods = pd.period_range(start=dataset["time_index"].min(),
                                           end=dataset["time_index"].max(),
                                           freq=resolution_str(self.sample_time_resolution))
        missing_periods = complete_periods[~complete_periods.isin(dataset["time_index"].values)]
        full_data = pd.concat([dataset[["time_index", "channel", "value"]],
                               pd.DataFrame({"time_index": missing_periods, "channel": -1, "value": 0})])
        del dataset
        full_data = full_data.sort_values(by="time_index")
        preprocessed_data = pd.DataFrame(index=full_data["time_index"].unique())
        for channel in full_data["channel"].unique():
            channel_full_data = full_data[full_data["channel"] == channel][["time_index", "value"]]
            preprocessed_data.loc[channel_full_data["time_index"], channel] = channel_full_data["value"].values
        preprocessed_data = preprocessed_data[np.sort(preprocessed_data.columns)].fillna(0)
        preprocessed_data = preprocessed_data[[c for c in preprocessed_data.columns if c != -1]]

        real_channels = preprocessed_data.columns
        del full_data
        real_time = preprocessed_data.index.values
        preprocessed_data = preprocessed_data.values

        return preprocessed_data, metadata, real_channels, real_time

    def _process(self, *args, **kwargs):
        raise NotImplementedError("Specific preprocessing classes must implement _process method")

    @staticmethod
    def process_predictions(predictions):
        return predictions


class SpikesCountPredictionScheme(PredictionScheme):
    def __init__(self, sample_time_resolution, min_isi=False):
        super().__init__(sample_time_resolution, min_isi)
        self.metric = BalancedRMSEMetric()
        self.baselines = [MeanBaseline()]
        self.name = "firing_rate"

    def _process(self, dataset):
        dataset = dataset.groupby(["channel", "time_index"]).agg(value=("time", "size"))
        dataset = dataset.reset_index()[["channel", "time_index", "value"]]
        return dataset


# Evaluation:
class Metric:
    def __init__(self):
        pass


class BalancedRMSEMetric(Metric):
    def __init__(self):
        self.name = "B-RMSE"
        super().__init__()

    @staticmethod
    def _filtered_rmse(y_true, y_pred, condition):
        if np.sum(condition) == 0:
            return 0
        return np.sqrt(np.mean((y_pred[condition] - y_true[condition]) ** 2))

    def compute(self, y_true, y_pred):
        tp = self._filtered_rmse(y_true, y_pred, (y_true != 0) & (y_pred != 0))
        fp = self._filtered_rmse(y_true, y_pred, (y_true == 0) & (y_pred != 0))
        # tn = self._filtered_rmse(y_true, y_pred, (y_true == 0) & (y_pred == 0))
        fn = self._filtered_rmse(y_true, y_pred, (y_true != 0) & (y_pred == 0))
        return (tp + fp + fn) / 4


class MeanBaseline:
    def __init__(self):
        self.baseline_name = "Mean"

    @staticmethod
    def predict(y_train_orig, y_test):
        return np.mean(y_train_orig) * np.ones(y_test.shape)


# Models:
TRAIN_SIZE = 0.8
class BasePredictor:
    def __init__(self):
        pass


class RegressionPredictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self._model = self._model_class()

    def reset_model(self):
        self._model = self._model_class()

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def prediction_params(self):
        return self._model.coef_

    def intercept(self):
        return self._model.intercept_


class SingleHistoryRegularPredictor(RegressionPredictor):
    def __init__(self, lags, horizons):
        self.model_name = "single_neuron_history_regression (normal)"
        self._lags = lags
        self._horizons = horizons
        self.hyperparams = list(product(lags, horizons))
        self._model_class = LinearRegression
        self.features_based = False
        super().__init__()

    def initiate_prediction_params(self, data, channels):
        params = {}
        for hyperparam in self.hyperparams:
            params[hyperparam] = np.zeros((0, hyperparam[0]))
        return params

    @staticmethod
    def train_test_data(data, channel_i, channels, hyperparam):
        l, h = hyperparam[0], hyperparam[1]
        channel_data = np.zeros((data.shape[0] - (l + h), (l + h)))
        for sample_i in range(channel_data.shape[0]):
            channel_data[sample_i, :] = data[sample_i:(sample_i + l + h), channel_i]
        train_data, test_data = train_test_split(channel_data, train_size=TRAIN_SIZE, shuffle=False)
        X_train, y_train = train_data[:, :-h], train_data[:, -1]
        X_test, y_test = test_data[:, :-h], test_data[:, -1]
        return X_train, X_test, y_train, y_test


class SingleHistorySGDPredictor(RegressionPredictor):
    def __init__(self, lags, horizons):
        self.model_name = "single_neuron_history_regression (SGD)"
        self._lags = lags
        self._horizons = horizons
        self.hyperparams = list(product(lags, horizons))
        self._model_class = SGDRegressor
        self.features_based = False
        super().__init__()

    def initiate_prediction_params(self, data, channels):
        params = {}
        for hyperparam in self.hyperparams:
            params[hyperparam] = np.zeros((0, hyperparam[0]))
        return params

    @staticmethod
    def train_test_data(data, channel_i, channels, hyperparam):
        l, h = hyperparam[0], hyperparam[1]
        channel_data = np.zeros((data.shape[0] - (l + h), (l + h)))
        for sample_i in range(channel_data.shape[0]):
            channel_data[sample_i, :] = data[sample_i:(sample_i + l + h), channel_i]
        train_data, test_data = train_test_split(channel_data, train_size=TRAIN_SIZE, shuffle=False)
        X_train, y_train = train_data[:, :-h], train_data[:, -1]
        X_test, y_test = test_data[:, :-h], test_data[:, -1]
        return X_train, X_test, y_train, y_test


# Plot:
plt.style.use('ggplot')
class Plot:
    def __init__(self, fig=None, ax=None, title=None, show=True, title_fontsize=None):
        if fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
        else:
            self.fig = fig
        if ax is None:
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = ax
        if title is not None:
            if title_fontsize is not None:
                self.ax.set_title(title, fontsize=title_fontsize)
            else:
                self.ax.set_title(title)
        self.show = show

    def plot(self, *args, **kwargs):
        self._plot(*args, **kwargs)
        plt.tight_layout()
        if self.show:
            plt.show()

    def _plot(self, *args, **kwargs):
        raise NotImplementedError("Specific plotting classes must implement _plot method")

class ViolinPlot(Plot):
    def __init__(self, fig=None, ax=None, title=None, show=True):
        super().__init__(fig, ax, title, show)

    def _plot(self, metric, categories):
        parts = self.ax.violinplot([metric[c] for c in categories], showmeans=True)
        for partname, part in parts.items():
            if partname == "bodies":
                for i, body in enumerate(part):
                    body.set_facecolor(plt.colormaps["Set1"](i))
            else:
                part.set_edgecolor("black")
                part.set_linewidth(0.5)
        self.ax.set_xticks(np.arange(len(categories)) + 1, categories, fontsize=20, rotation=45, ha="right")

class MultilineWithMeanPlot(Plot):
    def __init__(self, fig=None, ax=None, title=None, show=False, title_fontsize=None):
        super().__init__(fig, ax, title, show, title_fontsize)

    def _plot(self, data, x_values=None):
        if x_values is None:
            x_values = np.arange(data.shape[1])
        for i in range(data.shape[0]):
            self.ax.plot(x_values, data[i, :], color="gray", alpha=0.1)
        self.ax.plot(x_values, data.mean(axis=0))


class SingleExperimentSingleHyperparamRegressionPlot(Plot):
    def __init__(self, fig=None, ax=None, title=None, show=True):
        self.fig, self.ax = fig, ax
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
        if self.ax is None:
            self.ax = [self.fig.add_subplot(121), self.fig.add_subplot(122)]
        super().__init__(self.fig, self.ax, title, show)

    def _plot(self, metrics_dict, regression_coefs, results_title):
        ViolinPlot(fig=self.fig, ax=self.ax[0], show=False).plot(metrics_dict, metrics_dict.keys())
        x_values = np.arange(-regression_coefs.shape[1], 0)
        MultilineWithMeanPlot(fig=self.fig, ax=self.ax[1], show=False).plot(regression_coefs,
                                                                            x_values=x_values)
        plt.suptitle(results_title)
