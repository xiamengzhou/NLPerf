from sklearn import gaussian_process
from functools import partial
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from gp import run_gp_train, run_gp_test, ExactGPModel
from utils import recover
import gpytorch

def train_regressor(train_feats, train_labels, regressor="xgboost", quantile=0.95, paras=None):
    if regressor == "xgboost":
        reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate=0.2,
                               max_depth=5, n_estimators=200)
    elif regressor == "gp":
        kernel=1**2 + Matern(length_scale=2, nu=1.5) + WhiteKernel(noise_level=1)
        reg = gaussian_process.GaussianProcessRegressor(alpha=1e-10, copy_X_train=True, kernel=kernel,
                                                        n_restarts_optimizer=0, normalize_y=False,
                                                        optimizer='fmin_l_bfgs_b', random_state=None)
    elif regressor == "lower_xgbq":
        reg = XGBQuantile(n_estimators=200, max_depth=5)
        reg.set_params(quant_alpha=1. - quantile, quant_delta=1.0, quant_thres=5.0, quant_var=3.2)
    elif regressor == "upper_xgbq":
        reg = XGBQuantile(n_estimators=200, max_depth=5)
        reg.set_params(quant_alpha=quantile, quant_delta=1.0, quant_thres=6.0, quant_var=4.2)
    elif regressor == "lower_gb":
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                        learning_rate=.1, min_samples_leaf=9,
                                        min_samples_split=9)
        reg.set_params(loss='quantile', alpha=1.-quantile)
    elif regressor == "upper_gb":
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                        learning_rate=.1, min_samples_leaf=9,
                                        min_samples_split=9)
        reg.set_params(loss='quantile', alpha=quantile)
    elif regressor == "gb":
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                        learning_rate=.1, min_samples_leaf=9,
                                        min_samples_split=9)
    elif regressor == "gpytorch":
        assert paras is not None
        mean_module = paras["mean_module"]; covar_module = paras["covar_module"]
        reg = run_gp_train(train_feats, train_labels, mean_module, covar_module)
    else:
        print("Please specify a valid regressor!")
        return
    if len(reg) == 1: # not exactgp model
        fit_regressor(reg, train_feats, train_labels)
    return reg

def fit_regressor(reg, train_feats, train_labels):
    reg.fit(train_feats, train_labels)
    return reg

def get_valid_index(preds, labels):
    # legacy
    preds = np.where(preds == None, np.nan, preds)
    labels = np.where(labels == None, np.nan, labels)
    return np.intersect1d(np.argwhere(~pd.isnull(labels)), np.argwhere(~pd.isnull(preds)))

def calculate_rmse(preds, labels):
    valid_index = get_valid_index(preds, labels)
    return np.sqrt(mean_squared_error(labels[valid_index], preds[valid_index]))

def calculate_mean_bounds(lower_preds, upper_preds):
    valid_index = get_valid_index(lower_preds, upper_preds)
    return np.mean(upper_preds[valid_index] - lower_preds[valid_index])

# modify the function to get confidence band
def test_regressor(reg, test_feats, test_labels=None, get_rmse=True,
                   get_ci=False, quantile=0.95, lower_reg=None, upper_reg=None,
                   mns=None, sstd=None):
    preds = None; lower_preds = None; upper_preds = None; rmse = None
    # print(len(reg) == 2)
    # print(isinstance(reg[0], ExactGPModel))
    if get_ci:
        if isinstance(reg, xgb.XGBRegressor):
            preds = reg.predict(test_feats)
            lower_preds = lower_reg.predict(test_feats)
            upper_preds = upper_reg.predict(test_feats)
        elif isinstance(reg, gaussian_process.GaussianProcessRegressor):
            preds, std = reg.predict(test_feats, return_std=True)
            i = norm.ppf((1 - quantile) / 2)
            lower_preds, upper_preds = preds + i*std, preds - i*std
        elif isinstance(reg, GradientBoostingRegressor):
            preds = reg.predict(test_feats)
            lower_preds = lower_reg.predict(test_feats)
            upper_preds = upper_reg.predict(test_feats)
        elif len(reg) == 2 and isinstance(reg[0], ExactGPModel):
            # one percent of the times there are bugs
            preds, lower_preds, upper_preds = run_gp_test(reg, test_feats)
        else:
            preds = reg.predict(test_feats)
            print("Confidence band not supported for {}.".format(type(reg)))
    else:
        if len(reg) == 1:
            preds = reg.predict(test_feats)
        else:
            preds, _, _ = run_gp_test(reg, test_feats)
    if mns is not None and sstd is not None: # recover standardization
        preds = recover(mns, sstd, preds)
        test_labels = recover(mns, sstd, test_labels)
        if get_ci:
            lower_preds = recover(mns, sstd, lower_preds)
            upper_preds = recover(mns, sstd, upper_preds)
    if get_rmse:
        rmse = calculate_rmse(test_labels, preds)
    return preds, lower_preds, upper_preds, rmse

# Copy from https://towardsdatascience.com/regression-prediction-intervals-with-xgboost-428e0a018b
# quantile regression for xgboost
#@title XGBQuantile Class
class XGBQuantile(xgb.XGBRegressor):
    def __init__(self,quant_alpha=0.95,quant_delta = 1.0,quant_thres=1.0,quant_var =1.0,base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                 n_jobs=1, nthread=None, objective='reg:linear', random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,silent=True, subsample=1):
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var

        super().__init__(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
                         colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate, max_delta_step=max_delta_step,
                         max_depth=max_depth, min_child_weight=min_child_weight, missing=missing, n_estimators=n_estimators,
                         n_jobs= n_jobs, nthread=nthread, objective=objective, random_state=random_state,
                         reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, seed=seed,
                         silent=silent, subsample=subsample)

        self.test = None

    def fit(self, X, y):
        super().set_params(objective=partial(XGBQuantile.quantile_loss,alpha = self.quant_alpha,delta = self.quant_delta,threshold = self.quant_thres,var = self.quant_var) )
        super().fit(X,y)
        return self

    def predict(self,X):
        return super().predict(X)

    def score(self, X, y):
        y_pred = super().predict(X)
        score = XGBQuantile.quantile_score(y, y_pred, self.quant_alpha)
        score = 1./score
        return score

    @staticmethod
    def quantile_loss(y_true,y_pred,alpha,delta,threshold,var):
        x = y_true - y_pred
        grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-  ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
        hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta

        grad = (np.abs(x)<threshold )*grad - (np.abs(x)>=threshold )*(2*np.random.randint(2, size=len(y_true)) -1.0)*var
        hess = (np.abs(x)<threshold )*hess + (np.abs(x)>=threshold )
        return grad, hess

    @staticmethod
    def original_quantile_loss(y_true,y_pred,alpha,delta):
        x = y_true - y_pred
        grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
        hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta
        return grad,hess

    @staticmethod
    def quantile_score(y_true, y_pred, alpha):
        score = XGBQuantile.quantile_cost(x=y_true-y_pred,alpha=alpha)
        score = np.sum(score)
        return score

    @staticmethod
    def quantile_cost(x, alpha):
        return (alpha-1.0)*x*(x<0)+alpha*x*(x>=0)

    @staticmethod
    def get_split_gain(gradient,hessian,l=1):
        split_gain = list()
        for i in range(gradient.shape[0]):
            split_gain.append(np.sum(gradient[:i])/(np.sum(hessian[:i])+l)+np.sum(gradient[i:])/(np.sum(hessian[i:])+l)-np.sum(gradient)/(np.sum(hessian)+l) )

        return np.array(split_gain)

