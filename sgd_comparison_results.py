import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sgd_comparison_results_utils import CacheReader
from sgd_comparison_results_utils import SingleExperimentSingleHyperparamRegressionPlot
from sgd_comparison_results_utils import SpikesCountPredictionScheme, SingleHistoryRegularPredictor, \
    SingleHistorySGDPredictor

SAMPLE_TIME_RESOLUTION = 0.01


def multichannel_activity_prediction(prediction_scheme, model, data, channels, experiment):
    metrics = {}
    prediction_params = model.initiate_prediction_params(data, channels)
    for hyperparam in model.hyperparams:
        desc = f"{experiment} {model.model_name} | hyperparams: {hyperparam}"
        print(desc)
        metrics[hyperparam], prediction_params[hyperparam] = single_hp_predict(prediction_scheme, model,
                                                                               data,
                                                                               channels, hyperparam,
                                                                               prediction_params[hyperparam],
                                                                               desc)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        SingleExperimentSingleHyperparamRegressionPlot(fig=fig, ax=ax, show=True).plot(metrics[hyperparam],
                                                                                       prediction_params[hyperparam],
                                                                                       desc)
    return metrics, prediction_params


def single_hp_predict(prediction_scheme, model, data, channels, hyperparam,
                      hp_prediction_params, desc):
    hp_metrics = {}
    intercepts = []
    for channel_i, channel in tqdm(enumerate(channels), total=len(channels), desc=desc):
        model.reset_model()
        X_train, X_test, y_train, y_test = model.train_test_data(data, channel_i, channels, hyperparam)

        train_positive_indices = np.where(y_train.astype(bool))[0]
        if np.where(~y_train.astype(bool))[0].shape <= y_train.astype(bool).sum():
            train_null_indices = np.where(~y_train.astype(bool))[0]
        else:
            train_null_indices = np.random.choice(np.where(~y_train.astype(bool))[0], y_train.astype(bool).sum(),
                                                  replace=False)

        train_indices = np.sort(np.concatenate([train_positive_indices, train_null_indices]))
        X_train = X_train[train_indices, :]
        y_train_orig = y_train.copy()
        y_train = y_train[train_indices]

        for baseline_name in ["data", "surrogate"] + [b.baseline_name for b in prediction_scheme.baselines]:
            if baseline_name not in hp_metrics.keys():
                hp_metrics[baseline_name] = np.array([])

        for data_type in ("data", "surrogate"):
            if data_type == "surrogate":
                data_indices = np.zeros(X_train.shape)
                for i in range(X_train.shape[0]):
                    data_indices[i, :] = np.random.choice(np.arange(X_train.shape[1]), X_train.shape[1],
                                                          replace=False)
            else:
                data_indices = np.tile(np.arange(X_train.shape[1])[:, None], X_train.shape[0]).T

            model.fit(X_train[np.arange(X_train.shape[0])[:, None], data_indices.astype(int)], y_train)

            if data_type == "data":
                hp_prediction_params = np.vstack([hp_prediction_params, model.prediction_params()])
                intercepts += [model.intercept()]

            y_test_pred = prediction_scheme.process_predictions(model.predict(X_test))

            hp_metrics[data_type] = np.concatenate([hp_metrics[data_type],
                                                    [prediction_scheme.metric.compute(y_test, y_test_pred)]])
        for baseline in prediction_scheme.baselines:
            baseline_metric = prediction_scheme.metric.compute(y_test, baseline.predict(y_train_orig, y_test))
            hp_metrics[baseline.baseline_name] = np.concatenate([hp_metrics[baseline.baseline_name],
                                                                 [baseline_metric]])
    return hp_metrics, hp_prediction_params


if __name__ == '__main__':
    data_reader = CacheReader()
    dataset = data_reader.read()
    sample_time_resolution = SAMPLE_TIME_RESOLUTION
    prediction_scheme = SpikesCountPredictionScheme(sample_time_resolution, min_isi=False)

    experiment = "21645_Jan28_2024__18"
    experiment_dataset = dataset[dataset["experiment"] == experiment]
    data, metadata, real_channels, real_time = prediction_scheme.process(experiment_dataset)

    lags = [100]
    horizons = [1]

    for model_class in [SingleHistoryRegularPredictor, SingleHistorySGDPredictor]:
        model = model_class(lags, horizons)
        multichannel_activity_prediction(prediction_scheme, model, data, real_channels, experiment)
