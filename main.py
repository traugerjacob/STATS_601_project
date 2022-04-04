# file to submit to the online judge
# Josh Wasserman (jwasserman2)
# April 2022

from datetime import datetime, date, timedelta
import pandas as pd
import pickle
import numpy as np

with open("mod.pkl", "rb") as f:
    MOD = pickle.load(f)


def get_r_hat(A, B):
    pred_idx = A.index[-1]
    pred_df = pd.DataFrame({
        "asset": np.arange(10),
        "weekday_"  + str(pred_idx.day_of_week): np.ones(10),
        "log_vol_sum": np.log(B.loc[pred_idx - timedelta(minutes=MOD.minute_lag):pred_idx].sum(0)),
        "interval_high": A.loc[pred_idx - timedelta(minutes=MOD.minute_lag):pred_idx].max(0),
        "interval_low": A.loc[pred_idx - timedelta(minutes=MOD.minute_lag):pred_idx].min(0),
        "rsi": A.loc[pred_idx - timedelta(minutes=MOD.rsi_k):pred_idx].applymap(lambda x: max(x, 0)).sum(0) /
        (A.loc[pred_idx - timedelta(minutes=MOD.rsi_k):pred_idx].applymap(lambda x: max(x, 0)).sum(0) +
         A.loc[pred_idx - timedelta(minutes=MOD.rsi_k):pred_idx].applymap(lambda x: max(-x, 0)).sum(0)),
    })
    pred_df = pred_df.assign(**{"weekday_" + str(c): np.zeros(10) for c in range(7) if c != pred_idx.day_of_week})
    pred_df["rel_price_range"] = 2 * (pred_df["interval_high"] - pred_df["interval_low"]) / (
        pred_df["interval_high"] + pred_df["interval_low"])
    pred_df["range_volatility"] = np.sqrt(
        np.square(np.log(np.exp(pred_df["interval_high"]) / np.exp(pred_df["interval_low"]))) / (4 * np.log(2))
    )
    pred_df = pd.concat([
        pred_df,
        pd.concat([
        pd.concat([A.shift(ell).loc[pred_idx]] * 10, axis=1).set_axis(["asset_" + str(k) + "_lag_" + str(ell) for k in range(10)], axis=1)
        for ell in range(1, MOD.minute_lag + 1)], axis=1)],
        axis=1)
    pred_df = pd.get_dummies(pred_df, columns=["asset"], prefix="asset").reset_index(drop=True)
    return MOD.predict(pred_df[MOD.regressor_cols])
