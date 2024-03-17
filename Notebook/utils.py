from AOE.utils import verbose_print
from AOE.bandit_plots import hit_ratio_analysis, reward_distribution_analysis, regret_plots
from AOE.gp_bandit_finance import gp_bandit_finance
from AOE.mtgp_bandit_finance import mtgp_bandit_finance
from AOE.plots import rescale_plot

import numpy as np, pandas as pd
import torch
import gpytorch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as md


def get_bandit_objects(strategies):
    # Likelihood models & non stationarity params
    likelihood              = gpytorch.likelihoods.GaussianLikelihood()
    size_window             = 50 # non stationarity tests (nb of points)
    bandit_params = {'size_window' : size_window,
                      'lambda'      : 0.6,
                      'delta_I'     : 0.5} # Delta is bound on type 1 error
    
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



def plot_distribution_statistic(bandit_name, asset_name, all_lrs, all_delta_II, all_lengthscales, all_taus, d_correls):
    rescale_plot(W = 7, 
                 l = 10, 
                 w = 3, 
                 fontsize = 10)

    fig, axes = plt.subplots(1, 4, constrained_layout=True)

    qnt99   = np.quantile(all_lrs, 0.99) 
    qnt02   = np.quantile(all_lrs, 0.01) 
    all_lrs_ = [s for s in all_lrs if s < qnt99]
    all_lrs_ = [s for s in all_lrs_ if s > qnt02]

    qnt99   = np.quantile(all_taus, 0.99) 
    qnt02   = np.quantile(all_taus, 0.01) 
    all_taus_ = [s for s in all_taus if s < qnt99]
    all_taus_ = [s for s in all_taus_ if s > qnt02]

    sns.histplot(all_lrs_, 
                 color = "k", 
                 label = "Compact", 
                 ax    = axes[0], lw=2, 
                 line_kws = dict(color='k', lw=3),
                 kde   = False, stat='density') #, binwidth=1.
    sns.histplot(all_delta_II, 
                 color = "k", 
                 label = "Compact", 
                 ax    = axes[1], lw=2, 
                 line_kws = dict(color='k', lw=3),
                 kde   = False, stat='density')

    all_lengthscales = pd.Series(all_lengthscales)[pd.Series(all_lengthscales)<np.quantile(all_lengthscales, 0.75)]

    sns.histplot(all_lengthscales, 
                 color = "k", 
                 label = "Compact", 
                 ax    = axes[2], lw=3, 
                 line_kws = dict(color='k', lw=3),
                 kde   = False, stat='density', alpha=0.7)
    correl_series = pd.Series(d_correls[bandit_name]).rolling(10).mean()
    sns.histplot(correl_series, 
                 color = "k", 
                 label = "Compact", 
                 ax    = axes[3], lw=3, 
                 line_kws = dict(color='k', lw=3),
                 kde   = False, stat='density', bins=10, alpha=0.7)


    axes[0].axvline(np.mean(all_taus_), color='blue', lw=3, alpha=0.7)
    axes[0].xaxis.set_major_formatter( mtick.StrMethodFormatter('${x:,.0f}$'))
    axes[0].legend([r'$\mathcal{C}_\text{I}$', '$\mathcal{R}$'],  loc='upper right', 
                   fancybox=True, framealpha=0.2, handlelength=0.4, ncol=1)


    axes[1].legend([r'$\delta_\text{II}$'],  loc='upper right', fancybox=True, framealpha=0.2, handlelength=0.4, ncol=2)
    axes[1].set_xticks(axes[1].get_xticks()[::2])
    axes[1].set_xlim(0, 1)

    axes[2].grid('both')
    axes[2].yaxis.tick_right()
    axes[2].yaxis.set_label_position("left")
    axes[2].set_axisbelow(True)
    axes[2].legend([r'$l$'],  
                      loc        = 'upper right', 
                      fancybox   = True, 
                      framealpha = 0.2, handlelength=0.4, ncol=2)

    axes[2].xaxis.set_major_formatter( mtick.StrMethodFormatter('${x:,.2f}$'))
    #ax.set_xticks(ax.get_xticks()[::1])
    #axes[2].set_xlim(-0.1, 10.)
    axes[3].set_xlim(-1, 1)
    axes[1].set_ylabel(''); axes[2].set_ylabel(''); axes[1].set_ylabel('')

    axes[3].legend(['Correlation'],  
                      loc        = 'upper left', 
                      fancybox   = True, 
                      framealpha = 0.2, handlelength=0.4, ncol=2)

    for ax in axes:
        ax.grid('both')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("left")
        ax.set_axisbelow(True)
    
    plt.savefig(f'statistics_{asset_name}.pdf', bbox_inches='tight')
    plt.show()
    return correl_series, all_lrs_, all_taus_