from datetime import datetime, date, timedelta
from joblib import delayed, Parallel, parallel_backend, wrap_non_picklable_objects
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from time import time

lp = pd.read_pickle("log_price.df")
vol = pd.read_pickle("volume_usd.df")

LOOKAHEAD_PERIODS = 30
TRAIN_ADVANCE = 10
MINUTE_LAG = 30
N_MINUTE_LAGS = 10
RSI_K = 30
WF_TRAIN_WINDOW = 5000
WF_TEST_WINDOW = 2500

## Ziwei code
dt = timedelta(days=1)
r_hat = pd.DataFrame(index=lp.index[LAG::10], columns=np.arange(10), dtype=np.float64)
def get_r_hat(A, B):
    """
        A: 1440-by-10 dataframe of log prices with columns log_pr_0, ... , log_pr_9
        B: 1440-by-10 dataframe of trading volumes with columns volu_0, ... , volu_9
        return: a numpy array of length 10, corresponding to the predictions for the forward 30-minutes returns of assets 0, 1, 2, ..., 9
    """

    return -(A.iloc[-1] - A.iloc[-30]).values # Use the negative 30-minutes backward log-returns to predict the 30-minutes forward log-r

for t in lp.index[LAG::10]: # compute the predictions every 10 minutes
    # r_hat.loc[t, :] = get_r_hat(lp.loc[(t - dt):t], vol.loc[(t - dt):t])
    r_hat.loc[t, :] = -(lp.loc[(t - dt):t].iloc[-1] - lp.loc[(t - dt):t].iloc[-30]).values
r_fwd = (lp.shift(-30) - lp).iloc[30::10].rename(columns={f"log_pr_{i}": i for i in range(10)})
r_fwd_all = r_fwd.iloc[:-3].values.ravel() # the final 3 rows are NaNs.
r_hat_all = r_hat.iloc[:-3].values.ravel()
np.corrcoef(r_fwd_all, r_hat_all)
##

def create_features(lp, vol):
    print("Making training df")
    start_time = time()
    full_train_df = pd.DataFrame()
    for j in lp:
        train_df = pd.DataFrame({
            "return": (lp[j].shift(-LOOKAHEAD_PERIODS) - lp[j])[LOOKAHEAD_PERIODS::TRAIN_ADVANCE], # resp variable
            "weekday":  lp.index[LOOKAHEAD_PERIODS::TRAIN_ADVANCE].day_of_week.astype(str), # day of the week
        }).dropna()

        log_vol_sum = []
        interval_high = []
        interval_low = []
        rsi = []
        for t in train_df.index:
            log_vol_sum.append(np.log(sum(vol.loc[t - timedelta(minutes=MINUTE_LAG):t][j]))) # log(volume sum) over (t - LAG):t
            interval_high.append(max(lp.loc[t - timedelta(minutes=MINUTE_LAG):t][j])) # max price over (t - LAG):t
            interval_low.append(min(lp.loc[t - timedelta(minutes=MINUTE_LAG):t][j])) # min price over (t - LAG):t
            rsi.append(
                sum([max(x, 0) for x in lp.loc[t - timedelta(minutes=RSI_K):t][j]]) /
                (sum([max(x, 0) for x in lp.loc[t - timedelta(minutes=RSI_K):t][j]]) +
                 sum([max(-x, 0) for x in lp.loc[t - timedelta(minutes=RSI_K):t][j]]))) # relative strength index

        train_df["log_vol_sum"] = log_vol_sum
        train_df["interval_high"] = interval_high
        train_df["interval_low"] = interval_low
        train_df["rsi"] = rsi
        train_df["rel_price_range"] = 2 * (train_df["interval_high"] - train_df["interval_low"]) / (
            train_df["interval_high"] + train_df["interval_low"]) # relative price range
        train_df["range_volatility"] = np.sqrt(
            np.square(np.log(np.exp(train_df["interval_high"]) / np.exp(train_df["interval_low"]))) / (4 * np.log(2))
        ) # parkinson's volatility

        for ell in range(1, MINUTE_LAG + 1):
            train_df["price_lag_" + str(ell)] = lp[j].shift(ell)[train_df.index]
            train_df["vw_price_lag_" + str(ell)] = (lp[j].shift(ell) * vol[j].shift(ell))[train_df.index]
            train_df = pd.concat([
                train_df,
                lp[[i for i in lp if i != j]].shift(ell).rename(
                    columns={k: "asset_" + str(k - 1) + "_lag_" + str(ell)
                             for k in lp[[i for i in lp if i != j]]}).loc[train_df.index]
            ], axis=1)
        full_train_df = pd.concat([full_train_df, train_df.reset_index()])

    duration = time() - start_time
    print("Finished making training df in %s seconds" % np.round(duration, 4))

    return full_train_df.reset_index(drop = True)


def walkforward_cv(data, y, regressor_cols, train_window, test_window, model_class, model_args):
    """
    Run walk-forward cross-validation in parallel for a given model with 'fit' and 'score'
    methods
    NOTE: need to order by increasing timestamp
    """
    @delayed
    @wrap_non_picklable_objects
    def _fit_and_score(train_X, test_X, model, regressor_cols, y):
        model.fit(train_X[regressor_cols], train_X[y])

        return np.corrcoef(model.predict(test_X[regressor_cols]), test_X[y])[0,1]

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
    model_jobs = [_fit_and_score(**b) for b in job_args.values()]
    print("Running jobs in parallel")
    with parallel_backend("multiprocessing"):
        model_scores = Parallel(n_jobs=min(nbatches, 20), verbose=11, pre_dispatch='n_jobs')(model_jobs)

    return model_scores


train_df = create_features(lp, vol)
train_df = train_df.sort_values("timestamp").dropna()
regressor_cols = [c for c in train_df.columns if c not in ["return", "timestamp"]]
model_scores = walkforward_cv(train_df, "return", regressor_cols, WF_TRAIN_WINDOW, WF_TEST_WINDOW,
                              RandomForestRegressor,
                              {'n_estimators': 50, 'max_depth': 20, 'max_features': 'auto', 'bootstrap': True})
model_scores = walkforward_cv(train_df, "return", regressor_cols, WF_TRAIN_WINDOW, WF_TEST_WINDOW,
                              RidgeCV, {"alphas": np.logspace(-1, 1)})
np.mean(model_scores)

SEG_LENGTH = 1
LAG = 30
EMA_M = 12
EMA_A = 2 / (1 + EMA_M)


# ema = np.zeros(btc.shape[0])
# emsd = np.zeros(btc.shape[0])
# for i in range(2, btc.shape[0] + 1):
#     idx = btc.shape[0] - i
#     ema[idx] = EMA_A * btc.iloc[idx]["return"] + (1 - EMA_A) * ema[idx + 1]
#     emsd[idx] = np.sqrt(EMA_A * np.square(btc.iloc[idx]["return"] - ema[idx + 1]) + (1 - EMA_A) * np.square(emsd[idx + 1]))
