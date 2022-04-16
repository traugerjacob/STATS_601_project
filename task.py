from adaboost import Adaboost
from ols import Lin_Reg 
from datetime import datetime, date, timedelta
from joblib import delayed, Parallel, parallel_backend, wrap_non_picklable_objects
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from time import time
from itertools import product
from statsmodels.regression.linear_model import OLS
from ols_tree import OLS_Tree

class Task():

    def __init__(self, data_path_lp, data_path_vol, pred_type=None, pred_args=None, assets_for_each_pred=np.arange(1, 11)):
        self.data_path_lp = data_path_lp
        self.data_path_vol = data_path_vol
        self.pred_types = pred_type
        self.pred_args = pred_args
        self.assets_per_pred = assets_for_each_pred
        self.set_predictors(pred_type)
        self.create_dataframes()

    def set_predictors(self, predictors):
        self.pred_types = predictors
        for i in range(len(self.pred_types)):
            if self.pred_types[i] == "adaboost":
                self.pred_types[i] = Adaboost
            elif self.pred_types[i] == "ols" or self.pred_types[i] == "OLS":
                self.pred_types[i] = Lin_Reg
            elif self.pred_types[i] == "ols_tree":
                self.pred_types[i] = OLS_Tree 
            # Add new elifs for new classes here as they are created

    def set_pred_args(self, pred_args):
        self.pred_args = pred_args

    def get_pred_args(self):
        return self.pred_args

    def save_models(self,):
        pass

    def load_models(path_to_models):
        pass

    def remove_outliers(self, data):
        reg_cols = [c for c in data if c not in ["return", "timestamp", "asset", "index"]]
        mod = OLS(data['return'], data[reg_cols])
        fit = mod.fit()
        influence = fit.get_influence()
        a, b = influence.cooks_distance
        keep_idx = np.where(a < 4 / len(a))[0]
        data = data.iloc[keep_idx]
        return data


    def hyperparams_search(self,parameters, asset, train_window=10000, test_window=1440):
        """Will run hyperparameter search on current prediction types. Will run through all possible combinations of the parameters.
        parameters = dictionary where key=name_of_parameter and value = array of all possible values you want it to take. 
        asset = integer that corresponds to which asset you want to run the hyperparameter search over.
        """
        print(parameters)
        key_arr = []
        prod_arr = []
        for key in parameters.keys():
            key_arr.append(key)
            prod_arr.append(parameters[key])
        best_corr = -10000
        best_params = {}
        for comb in product(*prod_arr):
            cur_args = {}
            for j in range(len(key_arr)):
                cur_args[key_arr[j]] = comb[j]
            self.pred_args[asset] = cur_args
            regressor_cols = [c for c in self.train_df_assets[asset] if c not in ["return", "timestamp", "asset", "index"]]
            y = "return"
            corr = self.walkforward_cv_one(self.train_df_assets_w_outliers[asset], y, regressor_cols, train_window, test_window, self.pred_types[asset], self.pred_args[asset])
            print(f"Asset {i} args: {cur_args} correlation: {corr}")
            if corr > best_corr:
                best_corr = corr
                best_params = cur_args
        return (best_corr, cur_args)

    def walkforward_cv(self, train_window=10000, test_window=1440, predictors=None, predictors_arguments=None):
        """Runs walkforward_cv on each model. Returns cv corr."""
        if predictors != None:
            self.set_predictors(predictors)
        if predictors_arguments != None:
            self.set_pred_args(predictors_arguments)
        regressor_cols = [c for c in self.train_df_assets[0] if c not in ["return", "timestamp", "asset", "index"]]
        y = "return"
        cors = np.zeros(len(self.pred_types))
        for i in range(len(self.pred_types)):
            assets = self.assets_per_pred[i]
            frames = [self.train_df_assets[j] for j in assets]
            dataset = pd.concat(frames)
            outlier_frames = [self.train_df_assets_w_outliers[j] for j in assets]
            outlier_dataset = pd.concat(outlier_frames)
            print(f"running {str(self.pred_types[i])} on asset(s) {assets}")
            cors[i] = self.walkforward_cv_one(outlier_dataset, y, regressor_cols, train_window, test_window, self.pred_types[i], self.pred_args[i])
            print(f"walkforward cv gives correlation of {cors[i]}")
        print(cors)
        return cors

    def train_models(self, predictors=None, predictors_arguments=None, names=None):
        if predictors != None:
            self.set_pred_types(predictors)
        if predictors_arguments != None:
            self.set_pred_args(predictors_arguments)
        self.models = []
        for i in range(len(predictors)):
            self.models.append(self.pred_types[i](self.pred_args[i]))
            assets = self.assets_per_pred[i]
            frames = [self.train_df_assets[j] for j in assets]
            dataset = pd.concat(frames)
            regressor_cols = [c for c in dataset if c not in ["return", "timestamp", "asset", "index"]]
            y = dataset["return"]
            self.models[i].fit(dataset[regressor_cols], y)
        return self.models


     
    def create_dataframes(self, train_advance=10, minute_lag=30, rsi_k=30):
        """
        Generate dataframe of features to use for prediction
        Params
        ------
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
        lp = pd.read_pickle(self.data_path_lp)
        vol = pd.read_pickle(self.data_path_vol)
        self.train_df_assets = {}
        self.train_df_assets_w_outliers = {}
        for i in range(10):
            outliers, no_outliers = self.create_datasets_one(i, lp, vol)
            self.train_df_assets[i] = no_outliers
            self.train_df_assets_w_outliers[i] = outliers
        return self.train_df_assets
    
    def create_datasets_one(self, asset, lp, vol, train_advance=10, minute_lag=90):
        """
        Generate dataframe of features to use for prediction
        Params
        ------
        * lp: DataFrame, log price data
        * vol: DataFrame, volume data
        * train_advance: int, number of data points to skip between data points that are kept for training or testing.
            Default is 10.
        * minute_lag: int, number of previous minutes to calculate lagged features over. Default is 30.
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
        #global lp
        #global vol
        #lp = lp.iloc[:,  asset]
        #vol = vol.loc[:,  asset]
        print("Making training df")
        start_time = time()
        start_row = max(minute_lag, 30)
        train_df = pd.DataFrame({
            "asset": str(asset), # asset number
            "return": (lp[asset].shift(-30) - lp[asset])[start_row::train_advance], # resp variable
            #"weekday":  lp.index[30::train_advance].day_of_week.astype(str), # day of the week
            }).iloc[:-int(30/train_advance)]
        # get log_vol_sum, interval_high, interval_low, and rsi for each increasing length time period up to minute_lag
        train_df = pd.concat([
            train_df,
            pd.concat([
                np.log(pd.concat([vol[asset].shift(i)[start_row::train_advance] for i in range(c)], axis=1).sum(1).rename(
                    "log_vol_sum_lag_" + str(c)))
                for c in range(1, minute_lag + 1)], axis=1).loc[train_df.index],
            pd.concat([
                pd.concat([lp[asset].shift(i)[start_row::train_advance] for i in range(c)], axis=1).max(1).rename(
                    "interval_high_lag_" + str(c))
                for c in range(1, minute_lag + 1)], axis=1).loc[train_df.index],
            pd.concat([
                pd.concat([lp[asset].shift(i)[start_row::train_advance] for i in range(c)], axis=1).min(1).rename(
                    "interval_low_lag_" + str(c))
                for c in range(1, minute_lag + 1)], axis=1).loc[train_df.index]
        ], axis=1)
        rsi_min_df = lp[asset].copy() * -1
        rsi_max_df = lp[asset].copy()
        rsi_min_df[rsi_min_df < 0] = 0
        rsi_max_df[rsi_max_df < 0] = 0
        #train_df = pd.concat([
        #    train_df,
        #    pd.concat([
        #        (pd.concat([rsi_max_df.shift(i)[start_row::train_advance] for i in range(c)], axis=1).sum(1) / (
        #            pd.concat([rsi_max_df.shift(i)[start_row::train_advance] for i in range(c)], axis=1).sum(1) +
        #            pd.concat([rsi_min_df.shift(i)[start_row::train_advance] for i in range(c)], axis=1).sum(1)
        #        )).rename("rsi_lag_" + str(c))
        #        for c in range(1, minute_lag + 1)], axis=1).loc[train_df.index]
        #], axis=1)
        #train_df = pd.concat([
        #    train_df,
        #    pd.concat([
        #        np.sqrt(
        #            np.square(np.log(np.exp(train_df["interval_high_lag_" + str(i)]) /
        #                             np.exp(train_df["interval_low_lag_" + str(i)]))) / (4 * np.log(2))
        #        ) for i in range(1, minute_lag + 1)], axis=1).set_axis(
        #        ["range_volatility_lag_" + str(i) for i in range(1, minute_lag + 1)], axis=1)
        #], axis=1)

        train_df = pd.concat([
            train_df,
            pd.concat([(lp - lp.shift(i))[start_row::train_advance].set_axis(
            ["return_" + str(o) + "_lag_" + str(i) for o in range(10)], axis=1).loc[train_df.index]
            for i in np.linspace(5, 30, 6, dtype=int)], axis=1)
        ], axis=1)
        out_train_df = train_df.reset_index().sort_values("timestamp").reset_index(drop=True)
        idx = train_df["asset"] == str(asset)
        train_df = train_df[idx].reset_index().sort_values("timestamp").reset_index(drop=True)
        train_df = self.remove_outliers(train_df).reset_index().sort_values("timestamp").reset_index(drop=True)
        duration = time() - start_time
        print("Finished making training df in %s seconds" % np.round(duration, 4))
        return (out_train_df.drop(columns=["asset"]), train_df.drop(columns=["asset"]))

    def walkforward_cv_one(self, data,
                       y,
                       features,
                       train_window,
                       test_window,
                       model_class,
                       model_args,
                       increasing_train_window=True,
                       return_individual_scores=False,
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
        regressor_cols = features
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
        if increasing_train_window:
            job_args = {
                b: {
                'train_X': self.remove_outliers(data).iloc[:(b * test_window + train_window)],
                'test_X': data.iloc[(b * test_window + train_window):((b+1) * test_window + train_window)],
                'model': model_class(model_args),
                'regressor_cols': regressor_cols,
                'y': y,
                }
                for b in range(nbatches)
            }
        else:
            job_args = {
                b: {
                'train_X': self.remove_outliers(data).iloc[(b * test_window):(b * test_window + train_window)],
                'test_X': data.iloc[(b * test_window + train_window):((b+1) * test_window + train_window)],
                'model': model_class(model_args),
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
            print("Running jobs sequentially")
            if return_individual_scores:
                model_scores = []
                for j in job_args.values():
                    model_scores.append(np.corrcoef(_fit_and_score(**j), rowvar=False)[0,1])
                return model_scores
            else:
                model_scores = pd.DataFrame()
                for j in job_args.values():
                    model_scores = pd.concat([model_scores, _fit_and_score(**j)], axis=0)
        return np.corrcoef(model_scores, rowvar=False)[0,1]

if __name__ == "__main__":
    #t = Task("log_price.df", "volume_usd.df", pred_type=["OLS" for i in range(4)], pred_args=[{},{},{},{}], assets_for_each_pred=[(0,), (1,4,5,6), (2,), (3,), (7,8,9)])
    t = Task("log_price.df", "volume_usd.df", pred_type=["OLS" for i in range(10)], pred_args=[{} for i in range(10)], assets_for_each_pred=[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)])
    t.walkforward_cv()
    
    #t = Task("log_price.df", "volume_usd.df", pred_type=["adaboost" for i in range(10)], pred_args=[{"n_estimators": 20} for i in range(10)], assets_for_each_pred=[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)])
    #t.hyperparams_search({"n_estimators": [1,2]},0)
    ##t.walkforward_cv()
    times = time()
    t = Task("log_price.df", "volume_usd.df", pred_type=["ols_tree" for i in range(10)], pred_args=[[.5, {}, {"n_estimators": 50}] for i in range(10)], assets_for_each_pred=[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)])
    t.walkforward_cv()
    #best_vals = []
    #for i in range(10):
    #    best_vals.append(t.hyperparams_search({"v":[i/20 for i in range(1, 20)], "n_estimators": [10,20,50]}, i))
    #print(best_vals)
    #print(f"time from start to finish: {times - time()}")
    #t.walkforward_cv()

    #t = Task("log_price.df", "volume_usd.df", pred_type=["adaboost" for i in range(10)], pred_args=[{"n_estimators": 10} for i in range(10)])
    #a = t.create_dataframes()
    #t.train_models(predictors=["adaboost" for i in range(10)], predictors_arguments=[{"n_estimators": 10} for i in range(10)])
    #t.walkforward_cv(predictors=)
    #params = {"n_estimators": [1,50]}
    #asset = 0
    #print(t.hyperparams_search(params, asset))
