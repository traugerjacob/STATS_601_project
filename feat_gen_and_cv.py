# Feature generation and walk-through CV code
# Josh Wasserman (jwasserman2)
# April 2022

from datetime import datetime, date, timedelta
from joblib import delayed, Parallel, parallel_backend, wrap_non_picklable_objects
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from time import time

def create_features(lp, vol, train_advance=10, minute_lag=30, rsi_k=30):
    """
    Generate dataframe of features to use for prediction

    Params
    ------
    * lp: DataFrame, log price data
    * vol: DataFrame, volume data
    * train_advance: int, number of data points to skip between data points that are kept for training or testing.
        Default is 10.
    * minute_lag: int, number of previous minutes to calculate lagged features over. Default is 30.
    * rsi_k: int, number of previous minutes over which to calculate relative strength index. Default is 30.

    Returns
    -------
    * DataFrame of shape (p * (n / train_advance), j_generated_features + 1) where
    n is nrows of lp and vol, p is number of assets (ncols) in lp and vol, and j_generated_features
    is the number of features generated by the function, which depends on the chosen parameter values
    (the first column of the dataframe, 'return', is the response variable)

    NOTE: the returned dataframe is ordered by asset with increasing timestamp,
    i.e. asset 1 and its features/response variable are the first n / train_advance rows, asset 2 and
    its features/response variable are the next n /train_advance rows, etc.
    """
    print("Making training df")
    start_time = time()
    full_train_df = pd.DataFrame()
    for j in lp:
        train_df = pd.DataFrame({
            "asset": str(j), # asset number
            "return": (lp[j].shift(-30) - lp[j])[30::train_advance], # resp variable
            "weekday":  lp.index[30::train_advance].day_of_week.astype(str), # day of the week
        }).dropna()
        train_df = pd.concat([train_df[["asset", "return"]],
                              pd.get_dummies(train_df["weekday"], prefix="weekday")], axis=1)

        log_vol_sum = []
        interval_high = []
        interval_low = []
        rsi = []
        for t in train_df.index:
            log_vol_sum.append(np.log(sum(vol.loc[t - timedelta(minutes=minute_lag):t][j]))) # log(volume sum) over (t - LAG):t
            interval_high.append(max(lp.loc[t - timedelta(minutes=minute_lag):t][j])) # max price over (t - LAG):t
            interval_low.append(min(lp.loc[t - timedelta(minutes=minute_lag):t][j])) # min price over (t - LAG):t
            rsi.append(
                sum([max(x, 0) for x in lp.loc[t - timedelta(minutes=rsi_k):t][j]]) /
                (sum([max(x, 0) for x in lp.loc[t - timedelta(minutes=rsi_k):t][j]]) +
                 sum([max(-x, 0) for x in lp.loc[t - timedelta(minutes=rsi_k):t][j]]))) # relative strength index

        train_df["log_vol_sum"] = log_vol_sum
        train_df["interval_high"] = interval_high
        train_df["interval_low"] = interval_low
        train_df["rsi"] = rsi
        train_df["rel_price_range"] = 2 * (train_df["interval_high"] - train_df["interval_low"]) / (
            train_df["interval_high"] + train_df["interval_low"]) # relative price range
        train_df["range_volatility"] = np.sqrt(
            np.square(np.log(np.exp(train_df["interval_high"]) / np.exp(train_df["interval_low"]))) / (4 * np.log(2))
        ) # parkinson's volatility

        for ell in range(1, minute_lag + 1):
            train_df = pd.concat([train_df, lp.shift(ell).rename(
                columns={k: "asset_" + str(k) + "_lag_" + str(ell) for k in lp}).loc[train_df.index]
            ], axis=1) # price of assets at time t - ell
            train_df["vw_price_lag_" + str(ell)] = (lp[j].shift(ell) * vol[j].shift(ell))[train_df.index] # volume-weighted price at time t - ell

        full_train_df = pd.concat([full_train_df, train_df.reset_index()])

    full_train_df = pd.concat([
        pd.get_dummies(full_train_df["asset"], prefix="asset"),
        full_train_df[[c for c in full_train_df if c != "asset"]]], axis=1)
    duration = time() - start_time
    print("Finished making training df in %s seconds" % np.round(duration, 4))

    return full_train_df.reset_index(drop = True)


def walkforward_cv(data,
                   y,
                   regressor_cols,
                   train_window,
                   test_window,
                   model_class,
                   model_args,
                   parallel=False):
    """
    Run walk-forward cross-validation in parallel for a given model with 'fit' and 'score'
    methods
    NOTE: need to order by increasing timestamp

    Params
    ------
    * data: DataFrame, includes both the response variable and features to use for training
    * y: str, name of the response column in `data`
    * regressor_cols: list, column names to use as features for training
    * train_window: int, number of samples to use for training in each walkforward-window
    * test_window: int, number of samples to use for validation in each walkforward-window
    * model_class: model class to use for training
    * model_args: dict, dictionary of the hyperparameters to use for the model architecture
    * parallel: bool, if True, will run jobs in parallel using a 'multiprocessing' backend. If False,
        will run jobs sequentially

    Returns
    -------
    Correlation coefficient between predictions and true values of the response over all
    of the test windows
    """
    def _fit_and_score(train_X, test_X, model, regressor_cols, y):
        model.fit(train_X[regressor_cols], train_X[y])

        return pd.DataFrame({
            "pred": model.predict(test_X[regressor_cols]),
            "y": test_X[y]})

    @delayed
    @wrap_non_picklable_objects
    def _delayed_fit_and_score(train_X, test_X, model, regressor_cols, y):
        return _fit_and_score(train_X, test_X, model, regressor_cols, y)


    nbatches = int(np.floor((data.shape[0] - train_window) / test_window))
    job_args = {
        b: {
        'train_X': data.iloc[(b * test_window):(b * test_window + train_window)],
        'test_X': data.iloc[(b * test_window + train_window):((b+1) * test_window + train_window)],
        'model': model_class(**model_args),
        'regressor_cols': regressor_cols,
        'y': y,
        }
        for b in range(nbatches)
    }
    if parallel:
        model_jobs = [_delayed_fit_and_score(**b) for b in job_args.values()]
        print("Running jobs in parallel")
        with parallel_backend("multiprocessing"):
            out = Parallel(n_jobs=min(nbatches, 20), verbose=11, pre_dispatch='n_jobs')(model_jobs)
        model_scores = pd.concat(out, axis=0)
    else:
        model_scores = pd.DataFrame()
        print("Running jobs sequentially")
        for j in job_args.values():
            model_scores = pd.concat([model_scores, _fit_and_score(**j)], axis=0)

    return np.corrcoef(model_scores, rowvar=False)[0,1]

if __name__ == "__main__":
    lp = pd.read_pickle("log_price.df")
    vol = pd.read_pickle("volume_usd.df")
    train_df = create_features(lp.iloc[0:10000], vol.iloc[0:10000])
    train_df = train_df.sort_values("timestamp").dropna()
    regressor_cols = [c for c in train_df.columns if c not in ["return", "timestamp"]]
    model_scores = walkforward_cv(train_df, "return", regressor_cols, 2000, 200, RidgeCV,
                   {"alphas": np.logspace(-1, 1)}, parallel=True)
