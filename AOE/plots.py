import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as md
import matplotlib.gridspec as gridspec
import numpy as np
from .stats import get_meta_order_details

fmt = '${x:,.2f}'
tick = mtick.StrMethodFormatter(fmt)
tick2 = mtick.StrMethodFormatter('${x:,.0f}')
normal = mtick.StrMethodFormatter('{x:,.0f}')
normal2 = mtick.StrMethodFormatter('{x:,.2f}')
percc = mtick.StrMethodFormatter('{x:,.0%}')

colors       = {'red': '#ff207c', 'grey': '#42535b', 'blue': '#207cff', 'orange': '#ffa320', 'green': '#00ec8b'}
config_ticks = {'size': 14, 'color': colors['grey'], 'labelcolor': colors['grey']}
config_title = {'size': 18, 'color': colors['grey'], 'ha': 'left', 'va': 'baseline'}


def plot_meta_order(order_df, LOB_features, feature_name):
    order_df_ = order_df.iloc[:-1]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

    ax1.plot(order_df_.execution_time, 
            order_df_.execution_price, color='k')
    ax2.plot(LOB_features.set_index('timestamp').loc[order_df_.execution_time][feature_name].index,
            LOB_features.set_index('timestamp').loc[order_df_.execution_time][feature_name],
            color = 'k')
    ax3.plot(order_df_.execution_time, order_df_.meta_order_inventory, color='k')
    ax3.plot(order_df_.execution_time, order_df_.TWAP_inventory, color='tan')
    
    ax4.plot(order_df_.execution_time, order_df_.meta_order_pnl, color='k')
    ax4.plot(order_df_.execution_time, order_df_.TWAP_pnl, color='tan')

    for ax in (ax1, ax2, ax3, ax4):
        ax.tick_params(axis='both', **config_ticks)
        ax.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', color='gainsboro', linestyle='-', linewidth=0.5)
        ax.yaxis.tick_right()

    ax1.yaxis.set_major_formatter('${x:,.2f}'); ax1.legend(['Execution prices'], fontsize=12)
    ax2.yaxis.set_major_formatter('{x:,.2f}'); ax2.legend([feature_name], fontsize=12)
    ax3.yaxis.set_major_formatter('{x:,.0f}'); ax3.legend(['Inventory', 'TWAP inventory'], fontsize=12)
    ax4.yaxis.set_major_formatter('${x:,.0f}'); ax4.legend(['PnL', 'TWAP PNL'], fontsize=12)

    ax4.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))

def rescale_plot(W=2.5):
    # W = 2.5    # Figure width in inches, approximately A4-width - 2*1.25in margin

    plt.rcParams.update({
        'figure.figsize': (W, W/(4/3)),     # 4:3 aspect ratio
        'font.size' : 8,                   # Set font size to 11pt
        'axes.labelsize': 8,               # -> axis labels
        'legend.fontsize': 8,              # -> legends
        'font.family': 'lmodern',
        'text.usetex': True,
        'text.latex.preamble': (            # LaTeX preamble
            r'\usepackage{lmodern}'
            # ... more packages if needed
        )
    })



def hit_ratio_analysis(_historical_strats, _bandits, _historical_oracle_strats, _W = 5.5):
    bandit_k = list(_bandits.keys())[0]
    nb_bandits = len(_bandits.keys())
    
    print('Global number of rewards for every bandit:', len(_historical_strats[bandit_k]))
    for _bandit_k in _bandits.keys():    
        print(f'Hit ratio {_bandit_k}:', round(100*np.sum([s1==s2 for (s1, s2) in zip(_historical_strats[_bandit_k], _historical_oracle_strats)])/len(_historical_strats[_bandit_k])), '%')
    
    # Selected strategies
    rescale_plot(W=_W)
    fig, axs = plt.subplots(1, nb_bandits)
    
    for (_bandit_k, ax) in zip(_bandits.keys(), np.ndarray.flatten(axs)):
        ax.plot(_historical_strats[_bandit_k], color='grey')
        ax.plot(_historical_oracle_strats, color='tan', alpha=0.5)
        ax.legend([_bandit_k, 'Oracle'])

    plt.tight_layout()




def reward_distribution_analysis(bandit_name, bandits, period_est, LOB_features, rewards_to_plot, historical_reward_times, 
                                 historical_strats, historical_oracle_strats, strategies, W = 9, figure_name = None):
    rescale_plot(W=W)
    bandit_k = bandit_name
    fig = plt.figure()
    gs  = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[1, 2, 1])

    ax2 = fig.add_subplot(gs[1,0])
    ax0 = fig.add_subplot(gs[0,0], sharex=ax2)
    ax1 = fig.add_subplot(gs[2,0], sharex=ax2)

    # Prices
    sub_mid_prices = LOB_features.mid_price
    sub_mid_prices = sub_mid_prices[((sub_mid_prices.index<=historical_reward_times[-1]) & 
                                     (sub_mid_prices.index>=historical_reward_times[0]))]

    price_time_index = np.linspace(0, 1, len(sub_mid_prices)) #sub_mid_prices.index
    time_index       = np.linspace(0, 1, len(rewards_to_plot)) #sub_mid_prices.index

    ax1.plot(time_index, historical_strats[bandit_k], color='grey')
    ax1.plot(time_index, historical_oracle_strats, color='tan', alpha=0.5)
    ax1.legend([bandit_k, 'Oracle'])

    ax0.plot(price_time_index, sub_mid_prices.values, color='k')

    ax2legend = []
    for (strat, clrr) in zip(strategies.keys(), ('k', 'tan', 'b', 'g', 'grey', 'pink', 'silver')):
        stratAvg = rewards_to_plot[strat].rolling(period_est).mean()
        ax2.plot(time_index, stratAvg, color=clrr)
        ax2legend += [strat]

    for (strat, clrr) in zip(strategies.keys(), ('k', 'tan', 'b')):
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
    





def regret_plots(strategies, historical_reward_times, regrets_to_plot, LOB_features, bandits, W=5.5, figure_name = None):
    rescale_plot(W=W)
    fig, (ax0, ax2) = plt.subplots(2, 1, sharex=True)

    # Prices
    sub_mid_prices = LOB_features.mid_price
    sub_mid_prices = sub_mid_prices[((sub_mid_prices.index<=historical_reward_times[-1]) & 
                                     (sub_mid_prices.index>=historical_reward_times[0]))]

    price_time_index = np.linspace(0, 1, len(sub_mid_prices)) 
    time_index       = np.linspace(0, 1, len(regrets_to_plot))

    ax0.plot(price_time_index, sub_mid_prices.values, color='k')

    for (bandit_k, clrr) in zip( bandits.keys(), ('tan', 'grey', 'pink')):
        ax2.plot(time_index, regrets_to_plot[bandit_k].cumsum(), color=clrr)

    for (strat, clrr) in zip(strategies.keys(), ('k', 'b', 'g')):
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




def analyze_meta_order(meta_order_id, tape_meta_orders, W=7.8, figure_name = None):

    rescale_plot(W=W)

    feature_name = tape_meta_orders[meta_order_id].strategy.feature_name
    strategy_name = tape_meta_orders[meta_order_id].strategy.strategy_name

    print('Feature name :', feature_name)
    print('Strategy type:', strategy_name)
    print('Initial Inventory:', tape_meta_orders[meta_order_id].initial_inventory)

    trade_date = str(tape_meta_orders[meta_order_id].t0).split('T')[0]
    order_df   = get_meta_order_details(tape_meta_orders[meta_order_id], trade_date)
    order_df_  = order_df.iloc[:-1]

    fig, (ax1, ax3, ax4) = plt.subplots(3, 1, sharex=True)

    ax1.plot(order_df_.execution_time, 
            order_df_.execution_price, color='k')
    ax1.set_ylabel('Prices')

    ax3.plot(order_df_.execution_time, order_df_.meta_order_inventory, color='k')
    ax3.plot(order_df_.execution_time, order_df_.TWAP_inventory, color='tan')
    ax3.set_ylabel('inventory')

    ax4.plot(order_df_.execution_time, order_df_.meta_order_pnl, color='k')
    ax4.plot(order_df_.execution_time, order_df_.TWAP_pnl, color='tan')
    ax4.set_ylabel('PnL')

    for ax in (ax1, ax3, ax4):
        ax.grid(axis='both', color='gainsboro', linestyle='-', linewidth=0.5)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("left")

    ax1.yaxis.set_major_formatter('\${x:,.2f}'); ax1.legend(['MSFT'], fancybox=True, framealpha=0.5,)
    ax3.yaxis.set_major_formatter('{x:,.0f}'); ax3.legend([feature_name, 'Twap'], fancybox=True, framealpha=0.5,)
    ax4.yaxis.set_major_formatter('\${x:,.0f}'); 
    ax4.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    ax4.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    if figure_name:
        plt.savefig(figure_name, bbox_inches='tight')
    plt.show()