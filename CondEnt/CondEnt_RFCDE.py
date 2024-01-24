import pandas as pd
import seaborn as sns
import rfcde
import numpy as np
import warnings
import statsmodels.api as sm
from statsmodels.tsa.ar_model import ar_select_order, AutoReg
from scipy.integrate import simpson
from sklearn.linear_model import LinearRegression
from skgrf.ensemble import GRFForestLocalLinearRegressor
from copulae import EmpiricalCopula 
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class multivariate_EWS():

    def __init__(self, l):
        
        self.l = l

    def __select_Xlag_order(self, X):
        selector = ar_select_order(X, 13, old_names=False)

        try:
            k = int(selector.ar_lags[:-1])
        except:
            k = 0
        
        return k
        
    def data_transform(self, df, target, conditions):

        X = df[target].values.reshape(-1,1)
        self.k = self.__select_Xlag_order(X)
        self.cutting_edge = max(self.k, self.l)

        Y_lag = {}

        for f in conditions:
            df_Y = pd.DataFrame()
            for j in range(1, self.l+1):
                df_Y[f'{f}_{str(j)}'] = df[f].shift(j)
            Y_lag[f] = df_Y

        return X, Y_lag

    def get_AR_residuals(self, X):

        AR =AutoReg(X, self.k, trend = 'ct').fit()
        X_preds = AR.predict()[self.cutting_edge:]
        X_real = X.reshape(1,-1)[0][self.cutting_edge:]

        return np.array(X_preds - X_real)
    

    def get_ranks(self, res, est_res):

        if res.shape[0] != 1:
            res = res.reshape(-1,1)

        joint_vals = np.concatenate((res, est_res), axis = 1)
        ranks = EmpiricalCopula.pobs(joint_vals).T

        return ranks[0].reshape(-1,1), ranks[1:].T
    

    def get_AR_predict(self, X):

        AR =AutoReg(X, self.k, trend = 'ct').fit()
        print(AR.summary())
        X_preds = AR.predict()

        return X_preds
    
    
    def get_estimated_residuals_linear(self,
                                        res,
                                        conditions,
                                        fit_type = 'separately'):
        
        LR = LinearRegression()

        if fit_type == 'separately':
            estimated_res = pd.DataFrame()

            for i in conditions.keys():
                cond_var = conditions[i][self.cutting_edge:].to_numpy()
                LR.fit(cond_var, res.reshape(1,-1)[0])
                estimated_res[f'epsilon_{i}'] = LR.predict(cond_var)

            return estimated_res.to_numpy()
        
        elif fit_type == 'jointly':
            joint_cond = pd.DataFrame()
            for i in conditions.keys():
                joint_cond = pd.concat((joint_cond, conditions[i][self.cutting_edge:]), axis = 1)
            LR.fit(joint_cond.to_numpy(), res)

            return LR.predict(joint_cond.to_numpy()).reshape(-1,1)

        else:
            raise Exception('Inaapropriate value for "fit_type" parameter.')
        


    def get_estimated_residuals_LLF(self,
                                    res,
                                    conditions, 
                                    fit_type = 'separately',
                                    n_estimators=100,
                                    ll_split_weight_penalty=False,
                                    ll_split_lambda=0.1,
                                    ll_split_variables=None,
                                    ll_split_cutoff=None,
                                    equalize_cluster_weights=False,
                                    sample_fraction=0.5,
                                    mtry=None,
                                    min_node_size=5,
                                    honesty=True,
                                    honesty_fraction=0.5,
                                    honesty_prune_leaves=True,
                                    alpha=0.05,
                                    imbalance_penalty=0,
                                    ci_group_size=2,
                                    n_jobs=-1,
                                    seed=42,
                                    enable_tree_details=False,
                                    ):

        LLF = GRFForestLocalLinearRegressor(    
                                                n_estimators = n_estimators,
                                                ll_split_weight_penalty = ll_split_weight_penalty,
                                                ll_split_lambda = ll_split_lambda,
                                                ll_split_variables = ll_split_variables,
                                                ll_split_cutoff = ll_split_cutoff,
                                                equalize_cluster_weights = equalize_cluster_weights,
                                                sample_fraction = sample_fraction,
                                                mtry = mtry,
                                                min_node_size = min_node_size,
                                                honesty = honesty,
                                                honesty_fraction = honesty_fraction,
                                                honesty_prune_leaves = honesty_prune_leaves,
                                                alpha = alpha,
                                                imbalance_penalty = imbalance_penalty,
                                                ci_group_size = ci_group_size,
                                                n_jobs = n_jobs,
                                                seed = seed,
                                                enable_tree_details = enable_tree_details
                                         )

        if fit_type == 'separately':
            estimated_res = pd.DataFrame()

            for i in conditions.keys():
                cond_var = conditions[i][self.cutting_edge:].to_numpy()
                LLF.fit(cond_var, res.reshape(1,-1)[0])
                estimated_res[f'epsilon_{i}'] = LLF.predict(cond_var)

            return estimated_res.to_numpy()
        
        elif fit_type == 'jointly':
            joint_cond = pd.DataFrame()
            for i in conditions.keys():
                joint_cond = pd.concat((joint_cond, conditions[i][self.cutting_edge:]), axis = 1)
            LLF.fit(joint_cond.to_numpy(), res)

            return LLF.predict(joint_cond.to_numpy()).reshape(-1,1)

        else:
            raise Exception('Inaapropriate value for "fit_type" parameter.')

    
    def Est_Cond_Entropy_linear(self,
                        df,
                        target, 
                        conditions, 
                        window_size = 100, 
                        step = 1, 
                        use_ranks = False,
                        n_grid = 1000, 
                        bandwidth ='auto',  
                        n_trees = 3000, 
                        mtry = 1, 
                        node_size = 10, 
                        n_basis = 15, 
                        min_loss_delta = 0.001,
                        fit_type = 'separately',
                        verbose = True):

        number_of_itertaions = (len(df) - window_size) // step

        cond_entropy = []

        self.res_dict = {}
        self.est_dict = {}
        self.cond_dens_dict = {}

        for i in range(number_of_itertaions):
            if i % 5 == 0 and verbose == True:
                print('estimated fold:', f'[{i*step}:{i*step + window_size}]')

            X_fold, conditon_fold = self.data_transform(df[i*step:i*step + window_size], target, conditions)
            resids = self.get_AR_residuals(X_fold)
            estimated_resids = self.get_estimated_residuals_linear(resids, conditon_fold, fit_type = fit_type)

            if use_ranks:
                resids, estimated_resids = self.get_ranks(resids, estimated_resids)

            self.res_dict[f'[{i*step}:{i*step + window_size}]'] = resids
            self.est_dict[f'[{i*step}:{i*step + window_size}]'] = estimated_resids

            gridstep = 2 * (np.max(resids) - np.min(resids)) / n_grid
            grid = np.arange(2*np.min(resids), 2*np.max(resids), gridstep)

            if bandwidth == 'auto':
                bandwidth = 1.06 * np.sqrt(np.var(resids))*(window_size) ** (-0.2)

            rand_forest_dens_est = rfcde.RFCDE(n_trees=n_trees, mtry=mtry, node_size=node_size, min_loss_delta=min_loss_delta, n_basis=n_basis)
            rand_forest_dens_est.train(estimated_resids, resids)
            cond_PDF = rand_forest_dens_est.predict(estimated_resids, grid, bandwidth)

            self.cond_dens_dict[f'[{i*step}:{i*step + window_size}]'] = cond_PDF
        
            pointwise_entropy = []

            for i in range(len(cond_PDF)):
                entr = -simpson(cond_PDF[i] * np.log(cond_PDF[i]), dx = gridstep / n_grid)
                pointwise_entropy.append(entr)

                est_ent = np.array(pointwise_entropy)

                sum_of_ent = np.sum(est_ent[~(np.isnan(est_ent))])

            cond_entropy.append(sum_of_ent)

        return cond_entropy
    
    
    def Est_Cond_Entropy_LLF(self,
                        df,
                        target, 
                        conditions, 
                        window_size = 100, 
                        step = 1, 
                        use_ranks = False,
                        n_grid = 1000, 
                        bandwidth ='auto',  
                        n_trees_RFCDE = 3000, 
                        mtry = 1, 
                        node_size = 10, 
                        n_basis = 15, 
                        min_loss_delta = 0.001,
                        fit_type = 'separately',
                        verbose = True,
                        n_trees_LLF =1000,
                        ll_split_weight_penalty=False,
                        ll_split_lambda=0.1,
                        ll_split_variables=None,
                        ll_split_cutoff=None,
                        equalize_cluster_weights=False,
                        sample_fraction=0.5,
                        min_node_size=5,
                        honesty=True,
                        honesty_fraction=0.5,
                        honesty_prune_leaves=True,
                        alpha=0.05,
                        imbalance_penalty=0,
                        ci_group_size=2,
                        n_jobs=-1,
                        seed=42,
                        enable_tree_details=False):

        number_of_itertaions = (len(df) - window_size) // step

        cond_entropy = []

        self.res_dict = {}
        self.est_dict = {}
        self.cond_dens_dict = {}

        for i in range(number_of_itertaions):
            if i % 5 == 0 and verbose == True:
                print('estimated fold:', f'[{i*step}:{i*step + window_size}]')
            X_fold, conditon_fold = self.data_transform(df[i*step:i*step + window_size], target, conditions)
            resids = self.get_AR_residuals(X_fold)

            estimated_resids = self.get_estimated_residuals_LLF(resids,
                                                                conditon_fold,
                                                                fit_type = fit_type,
                                                                n_estimators = n_trees_LLF,
                                                                ll_split_weight_penalty = ll_split_weight_penalty,
                                                                ll_split_lambda = ll_split_lambda,
                                                                ll_split_variables = ll_split_variables,
                                                                ll_split_cutoff = ll_split_cutoff,
                                                                equalize_cluster_weights = equalize_cluster_weights,
                                                                sample_fraction = sample_fraction,
                                                                mtry = mtry,
                                                                min_node_size = min_node_size,
                                                                honesty = honesty,
                                                                honesty_fraction = honesty_fraction,
                                                                honesty_prune_leaves = honesty_prune_leaves,
                                                                alpha = alpha,
                                                                imbalance_penalty = imbalance_penalty,
                                                                ci_group_size = ci_group_size,
                                                                n_jobs = n_jobs,
                                                                seed = seed,
                                                                enable_tree_details = enable_tree_details)
            
            if use_ranks:
                resids, estimated_resids = self.get_ranks(resids, estimated_resids)

            

            self.res_dict[f'[{i*step}:{i*step + window_size}]'] = resids
            self.est_dict[f'[{i*step}:{i*step + window_size}]'] = estimated_resids

            gridstep = 2 * (np.max(resids) - np.min(resids)) / n_grid
            grid = np.arange(2*np.min(resids), 2*np.max(resids), gridstep)

            if bandwidth == 'auto':
                bandwidth = 1.06 * np.sqrt(np.var(resids))*(window_size) ** (-0.2)

            rand_forest_dens_est = rfcde.RFCDE(n_trees=n_trees_RFCDE, mtry=mtry, node_size=node_size, min_loss_delta=min_loss_delta, n_basis=n_basis)
            rand_forest_dens_est.train(estimated_resids, resids)
            cond_PDF = rand_forest_dens_est.predict(estimated_resids, grid, bandwidth)

            self.cond_dens_dict[f'[{i*step}:{i*step + window_size}]'] = cond_PDF
        
            pointwise_entropy = []

            for i in range(len(cond_PDF)):
                entr = -simpson(cond_PDF[i] * np.log(cond_PDF[i]), dx = gridstep / n_grid)
                pointwise_entropy.append(entr)

                est_ent = np.array(pointwise_entropy)

                sum_of_ent = np.sum(est_ent[~(np.isnan(est_ent))])

            cond_entropy.append(sum_of_ent)

        return cond_entropy