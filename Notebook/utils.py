from AOE.utils import verbose_print
from AOE.bandit_plots import hit_ratio_analysis, reward_distribution_analysis, regret_plots
from AOE.gp_bandit_finance_old import gp_bandit_finance, Wasserstein_GP_mean#, Wasserstein_GP
from AOE.mtgp_bandit_finance import mtgp_bandit_finance
from AOE.plots import rescale_plot

import numpy as np, pandas as pd
import torch
import gpytorch

def get_bandit_objects(strategies):
    # Likelihood models & non stationarity params
    likelihood              = gpytorch.likelihoods.GaussianLikelihood()
    size_window             = 50 # non stationarity tests (nb of points)
    bandit_params = {'size_window' : size_window,
                      'threshold'   : 0.5,
                      'lambda'      : 0.6,
                      'delta_I'     : 0.5,
                      'check_type_II' : False} # Delta is bound on type 1 error
    
    # Bandit objects
    return {'RANDOM'             : gp_bandit_finance(strategies, bandit_algo='RANDOM', likelihood=likelihood, 
                                                     bandit_params = bandit_params),
            'GREEDY'             : gp_bandit_finance(strategies, bandit_algo='GREEDY', likelihood=likelihood, 
                                                     bandit_params = bandit_params),
            'UCB'                : gp_bandit_finance(strategies, bandit_algo='MAB_UCB', likelihood=likelihood, 
                                                     bandit_params = bandit_params),
            'GP'                 : gp_bandit_finance(strategies, bandit_algo='UCB_NS', likelihood=likelihood, 
                                                     bandit_params = bandit_params),
            'GP_LR'              : gp_bandit_finance(strategies, bandit_algo='UCB_LR', likelihood=likelihood, 
                                                     bandit_params = bandit_params, reinit=False),
            'MTGP_LR'            : mtgp_bandit_finance(strategies, bandit_algo   = 'UCB_LR', likelihood    = likelihood, 
                                               bandit_params = bandit_params, reinit = False)            
            }

def init_variables(all_data, strategies):
    bandits = get_bandit_objects( strategies)
    bandit_rewards = pd.DataFrame(index=all_data.index)
    for bandit_k in bandits.keys(): 
        bandit_rewards[bandit_k] = np.nan
    bandit_rewards['oracle'] = np.nan
    for strat in strategies.keys():
        bandit_rewards[strat] = np.nan

    batch_times = list(all_data.index)

    bandits_fin_info = {bandit_k:pd.DataFrame(index=all_data.index) for bandit_k in bandits.keys()}
    for bandit_k in bandits.keys(): 
        bandits_fin_info[bandit_k]['execPrice'] = np.nan
        bandits_fin_info[bandit_k]['S0']        = np.nan
        bandits_fin_info[bandit_k]['ST']        = np.nan
        bandits_fin_info[bandit_k]['twapPrice'] = np.nan
        bandits_fin_info[bandit_k]['best_strat'] = np.nan

    return bandit_rewards, bandits_fin_info, batch_times, bandits


def init_dictionaries(strategies):
    d_lrstats    = {}
    d_taus       = {}
    d_delta_IIs  = {}
    d_noises     = {}
    d_lengthscales = {}
    d_correls    = {}

    b_objs = get_bandit_objects(strategies)

    for bandit_name in b_objs.keys():
        if 'MTGP_' in bandit_name:
            d_lrstats[bandit_name]    = {}
            d_taus[bandit_name]       = {}
            d_delta_IIs[bandit_name]  = {}
            d_noises[bandit_name]        = {}
            d_lengthscales[bandit_name]  = {}
            d_correls[bandit_name] = []
            for strat in strategies.keys():
                d_lrstats[bandit_name][strat]   = []
                d_taus[bandit_name][strat]      = []
                d_delta_IIs[bandit_name][strat] = []
                d_noises[bandit_name][strat]       = []
                d_lengthscales[bandit_name][strat] = []
    return d_lrstats, d_taus, d_delta_IIs, d_noises, d_lengthscales, d_correls, b_objs