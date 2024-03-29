from AOE.utils import verbose_print
from AOE.optimal_trading import execute_strategy
from AOE.stats import get_meta_order_details
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as md
import matplotlib.gridspec as gridspec
import numpy as np
from .plots import rescale_plot


def hit_ratio_analysis(_rewards_to_plot, _oracle_rewards, _bandits, _W = 5.5):
    bandit_k = list(_bandits.keys())[0]
    nb_bandits = len(_bandits.keys())
    
    print('Global number of rewards for every bandit:', len(_rewards_to_plot[bandit_k]))
    for _bandit_k in _bandits.keys():    
        print(f'Hit ratio {_bandit_k}:', round(100*np.sum([s1==s2 for (s1, s2) in zip(_rewards_to_plot.dropna()[_bandit_k], _oracle_rewards)])/len(_rewards_to_plot.dropna()[_bandit_k])), '%')
    
    # Selected strategies
#    rescale_plot(W=_W)
#    fig, axs = plt.subplots(1, nb_bandits)
#    
#    for (_bandit_k, ax) in zip(_bandits.keys(), np.ndarray.flatten(axs)):
#        ax.plot(_rewards_to_plot[_bandit_k], color='grey')
#        ax.plot(_oracle_rewards, color='tan', alpha=0.5)
#        ax.legend([_bandit_k, 'Oracle'])
#
#    plt.tight_layout()


def reward_distribution_analysis(bandit_name, bandits, period_est, all_data, rewards_to_plot, bandit_rewards,
                                 strategies, W = 9, figure_name = None):
    rescale_plot(W=W)
    bandit_k = bandit_name
    fig = plt.figure()
    gs  = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[1, 2, 1])

    ax2 = fig.add_subplot(gs[1,0])
    ax0 = fig.add_subplot(gs[0,0], sharex=ax2)
    ax1 = fig.add_subplot(gs[2,0], sharex=ax2)

    # Prices
    sub_mid_prices = all_data.mid_price
    sub_mid_prices = sub_mid_prices[((sub_mid_prices.index<=rewards_to_plot.index[-1]) & 
                                     (sub_mid_prices.index>=rewards_to_plot.index[0]))]

    price_time_index = np.linspace(0, 1, len(sub_mid_prices)) #sub_mid_prices.index
    time_index       = np.linspace(0, 1, len(rewards_to_plot)) #sub_mid_prices.index

    ax1.plot(time_index, rewards_to_plot[bandit_k], color='grey')
    ax1.plot(time_index, bandit_rewards['oracle'], color='tan', alpha=0.5)
    ax1.legend([bandit_k, 'Oracle'])

    ax0.plot(price_time_index, sub_mid_prices.values, color='k')

    ax2legend = []
    for (strat, clrr) in zip(strategies.keys(), ('k', 'tan', 'b', 'g', 'grey', 'pink', 'silver')):
        stratAvg = rewards_to_plot[strat].rolling(period_est).mean()
        ax2.plot(time_index, stratAvg, color=clrr)
        ax2legend += [strat]

    for (strat, clrr) in zip(strategies.keys(), ('k', 'tan', 'b', 'g')):
        stratAvg = rewards_to_plot[strat].rolling(period_est).mean()
        stratStd = rewards_to_plot[strat].rolling(period_est).std()
        ax2.fill_between(time_index, stratAvg - stratStd, stratAvg + stratStd, alpha=0.5, color=clrr)

    ax2.legend( ax2legend, ncol=2, fancybox=True, framealpha=0.2, loc='lower right')

    for ax in (ax0, ax2):
        ax.grid(axis='both', color='gainsboro', linestyle='-', linewidth=0.5)
        ax.yaxis.tick_right()
        ax.yaxis.set_major_formatter('{x:,.1f}%')

    ax1.grid(axis='both', color='gainsboro', linestyle='-', linewidth=0.5)
    ax1.yaxis.tick_right()

    ax0.set_ylabel('prices')
    ax2.set_ylabel('Regret')

    ax2.yaxis.set_major_formatter('{x:,.2f}\%')
    ax0.yaxis.set_major_formatter('\${x:,.2f}')
    ax2.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    if figure_name:
        plt.savefig(figure_name, bbox_inches='tight')
    plt.show()
    

def regret_plots(strategies, regrets_to_plot, all_data, bandits, W=5.5, figure_name = None):
    rescale_plot(W=W)
    fig, (ax0, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Prices
    sub_mid_prices = all_data.mid_price
    sub_mid_prices = sub_mid_prices[((sub_mid_prices.index<=regrets_to_plot.index[-1]) & 
                                     (sub_mid_prices.index>=regrets_to_plot.index[0]))]

    price_time_index = np.linspace(0, 1, len(sub_mid_prices)) 
    time_index       = np.linspace(0, 1, len(regrets_to_plot))

    ax0.plot(price_time_index, sub_mid_prices.values, color='k')

    for (bandit_k, clrr) in zip( bandits.keys(), ('tan', 'grey', 'pink', 'g')):
        ax2.plot(time_index, regrets_to_plot[bandit_k].cumsum(), color=clrr)

    for (strat, clrr) in zip(strategies.keys(), ('k', 'b', 'red')):
        ax2.plot(time_index, regrets_to_plot[strat].cumsum(), color=clrr)

    ax2.legend(list(bandits.keys()) +  list(strategies.keys()))

    for ax in (ax0, ax2):
        ax.grid(axis='both', color='gainsboro', linestyle='-', linewidth=0.5)
        ax.yaxis.tick_right()

    ax0.yaxis.set_major_formatter('\${x:,.2f}')
    ax2.yaxis.set_major_formatter('{x:,.2f}\%')
    ax0.set_ylabel('prices')
    ax2.set_ylabel('Regret')
    ax2.tick_params(axis='x', rotation=90)
    ax0.set_ylabel('prices')
    ax2.set_ylabel('Regret')
    ax2.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    
    if figure_name:
        plt.savefig(figure_name, bbox_inches='tight')
        
    plt.show()