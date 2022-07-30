import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from   IPython.display import display, HTML
from collections import deque
import random
import os

from AOE.utils import get_LOB_data, get_features, verbose_print
from AOE.stats import get_meta_order_df, get_meta_order_details

from AOE.order import order
from AOE.meta_order import meta_order
from AOE.strategy import strategy

import itertools
import multiprocessing as mp

import statsmodels.api as sm
import scipy.integrate as integrate

from scipy.interpolate import RegularGridInterpolator

import time
import pickle

def estimate_ou_params(feature_data, dn): # dt is in trade time
    y = feature_data.shift(-dn).iloc[:-dn].values
    x   = feature_data.iloc[:-dn].values

    model   = sm.OLS(y, x)
    results = model.fit()

    r      = (1-results.params[0])/dn
    sigmaI = results.resid.std() / dn**0.5

    return r, sigmaI, results.rsquared

def v2(t, phi, kappa, alpha, T):
    beta = np.sqrt(phi/kappa)
    xi   = (alpha + np.sqrt(kappa*phi)) / (alpha - np.sqrt(kappa*phi))
    return np.sqrt(kappa*phi) * (1 + xi*np.exp(2*beta*(T-t))) / (1 - xi*np.exp(2*beta*(T-t)))

def getOptimalSpeed2(qt, It, t, T, phi, kappa, alpha, thetaI, sigmaI):
    int_v2t = lambda s: integrate.quad(lambda t:v2(t, phi, kappa, alpha, T), t, s)[0]
    to_integrate = lambda s: np.exp(-thetaI * (s-t) + int_v2t(s) /kappa)
    optimalspeed = integrate.quad(to_integrate, t, T)[0]
    return -(optimalspeed*It + 2*v2(t, phi, kappa, alpha, T)*qt)/(2*kappa)



def sim(params):
    ## Parameter unpacking 
    strat = params[0]
    dyn = params[1]



    ## Data loading 
    data_path = os.getcwd() + f'/data/Market/S{dyn}'
    results_path = os.getcwd() + '/data/Rewards/S{dyn}'

    LOB_features      = pd.read_csv(f'{data_path}/simdata.csv', 
                                    engine    = 'python',
                                    index_col = 0,
                                    infer_datetime_format=True)
    LOB_features['timestamp'] = pd.to_datetime(LOB_features.timestamp.values)

    trade_date       = '2014-09-24'

    ###################
    # control variables
    ###################
    # arrival_intensity   = 0.01 # 100 trades every 30min
    meta_order_size     = 100
    verbose_level       = 1
    latency             = 0  # TODO: implement this
    T                   = 300 # in seconds
    smallest_order_size = 1  # TODO: implement this

    ###################
    # Finance variables
    ###################
    # Populating values
    if strat == 'IMBALANCE':
        features          = ['I1']
        strategies        = {'I1' : {'name'   : 'imbalance',
                                    'params' : {'feature_name'     : 'I1',
                                                'estimation_period': '5min', 
                                                'alpha'            : 10, 
                                                'phi'              : 0.005, 
                                                'kappa'            : 1e-10}}}  
    elif strat == 'TREND':
        features          = ['I2']
        strategies        = {'I2'     : {'name'   : 'trend',
                                      'params' : {'feature_name'     : 'I2',
                                                  'estimation_period': '5min', 
                                                  'alpha'            : 10, 
                                                  'phi'              : 0.005, 
                                                  'kappa'            : 1e-10}}}
    else:
        print('STRATEGY NOT IMPLEMENTED.')
        return

    ###################
    # Sampling variables
    ###################
    trailing_period   = 3600 # for sampling the best strategy
    trading_frequency = 0.01  # in seconds
    rewards           = {i: [] for i in strategies.keys()}
    TWAP_rewards      = {i: [] for i in strategies.keys()}


    ###################
    # Exec algo stuff
    ###################
    tape_pending_meta_orders     = {}
    tape_pending_orders = deque(maxlen=None) # orders to execute at time step t

    nb_features         = len(features)
    times               = LOB_features.index.values
    meta_order_id_c     = 0
    order_id_c          = 0

    ######################################
    # Precompute strategy output
    ######################################
    nb_q_      = 20
    nb_I_      = 20
    nb_t_      = 10
    nb_thetaI_ = 20

    for strategy_id in strategies:
        All_qs = np.linspace(-meta_order_size, meta_order_size, nb_q_)
        All_Is = np.linspace(-1, 1, nb_I_)
        All_thetaIs = np.linspace(0.0001, 0.08, nb_thetaI_)
    
        All_ts = list(np.linspace(0, T, nb_t_))
        All_ts = All_ts[:-1] + list(np.linspace(All_ts[-2], T, nb_t_))[1:]
        All_ts = np.array(All_ts[:-1] + list(np.linspace(All_ts[-2], T, nb_t_))[1:])
        nb_t_ = 3 * nb_t_ - 4

        optimal_speed_grid = np.zeros((nb_t_, nb_q_, nb_I_, nb_thetaI_))

        for qt_ in range(nb_q_):
            for It_ in range(nb_I_):
                for tt_ in range(nb_t_):
                    for thetaIt_ in range(nb_thetaI_):
                        optimal_speed_grid[tt_, qt_, It_, thetaIt_] = \
                                    getOptimalSpeed2(qt  = All_qs[qt_], 
                                                     It  = All_Is[It_], 
                                                     t   = All_ts[tt_], 
                                                     T   = T/(60*60*24),
                                                     phi = strategies[strategy_id]['params']['phi'], 
                                                     kappa  = strategies[strategy_id]['params']['kappa'], 
                                                     alpha  = strategies[strategy_id]['params']['alpha'], 
                                                     thetaI = All_thetaIs[thetaIt_], 
                                                     sigmaI = np.nan)

        fn = RegularGridInterpolator( (All_ts, All_qs, All_Is, All_thetaIs), optimal_speed_grid)

        strategies[strategy_id]['interpolator'] = (fn, All_ts, All_qs, All_Is, All_thetaIs)

    ###################
    # Run the backtest
    ###################
    start_time            = time.time()
    last_trading_time     = times[1000]
    meta_orders_to_delete = []

    for i in range(1000, len(times)-1):
        t   = times[i]
        t_1 = times[i-1]
        
        b_meta_order_arrival = 1 #np.random.poisson(arrival_intensity)
        
        # only trade given trading frequency
        if (t - last_trading_time) > trading_frequency:
            last_trading_time =  t
            
            ###################
            # Get market information
            ###################
            feature_values = LOB_features.loc[t_1, features]
            St             = LOB_features.at[t, 'mid_price']
            bestbid        = LOB_features.at[t, 'bid_1']
            bestask        = LOB_features.at[t, 'ask_1']

            ######################################
            # Execute pending orders and update inventories
            ######################################
            for order_item in tape_pending_orders:
                if order_item.quantity<0: # hit the best bid
                    order_item.executed(t, bestbid)
                else: #hit the best ask
                    order_item.executed(t, bestask)

                verbose_print(verbose_level, t, trade_date, f'Executed order = {order_item.order_id}, meta order = {order_item.the_meta_order.meta_order_id}, with strategy = {order_item.the_meta_order.strategy_id}, with quantity: {order_item.quantity} at price: {St}. Remaining inventory = {order_item.the_meta_order.current_inventory}, Remaining time in seconds = {order_item.the_meta_order.T-t}.')

            # clear the pending orders
            tape_pending_orders.clear()

            ######################################
            # If meta-orders are finished
            # record the reward & delete from tape
            ######################################
            # delete pending orders to be deleted
            for meta_order_item_k in meta_orders_to_delete:
                verbose_print(verbose_level, t, trade_date, f'Deleting the meta order: {meta_order_item_k}')
                
                tape_pending_meta_orders[meta_order_item_k].strategy    =  None
                meta_order_item_k_id = tape_pending_meta_orders[meta_order_item_k].meta_order_id
                # pickle it first
                with open(f'{results_path}/meta_order_{meta_order_item_k_id}_{strat}.pkl',"wb") as f:
                    # delete strategy object for memory space purposes
                    pickle.dump(tape_pending_meta_orders[meta_order_item_k], f)

                # delete it next
                tape_pending_meta_orders.pop(meta_order_item_k)
                
            meta_orders_to_delete = []
            
            for meta_order_item_k in tape_pending_meta_orders.keys():
                meta_order_item = tape_pending_meta_orders[meta_order_item_k]

                meta_order_is_over, meta_order_remaining = meta_order_item.is_over(t)
                if meta_order_is_over:
                    if meta_order_remaining: # time is up, and inventory is still not 0, send the last orders

                        verbose_print(verbose_level, t, trade_date, f'Warning: meta order {meta_order_item.meta_order_id} finished but not completely executed. Remaining inventory = {meta_order_item.current_inventory}, Sending a finish up order.')

                        o_order_quantity = meta_order_item.current_inventory
                        o_order          = meta_order_item.generate_order(order_id_c, o_order_quantity, t, feature_values=feature_values)

                        if np.abs(o_order.quantity)>0:
                            verbose_print(verbose_level, t, trade_date, f'Adding final order to pending orders list. Meta order = {meta_order_item.meta_order_id}. Order = {o_order.order_id}. quantity = {o_order.quantity}')

                            tape_pending_orders.append(o_order)
                            order_id_c += 1

                    else: # meta order is executed entirely
                        verbose_print(verbose_level, t, trade_date, f'The meta-order {meta_order_item.meta_order_id} is finished, with pnl = {meta_order_item.current_pnl}')

                    rewards[meta_order_item.strategy.strategy_id] += [meta_order_item.current_pnl]
                    meta_orders_to_delete += [meta_order_item_k]

            

            ######################################
            # If a new meta_order had arrived
            # create the objects
            ######################################    
            if b_meta_order_arrival>0:

                # create the meta order object
                o_meta_order  = meta_order(meta_order_id     = meta_order_id_c,             
                                        S0                = St,
                                        quantity          = b_meta_order_arrival*meta_order_size, 
                                        T                 = T,
                                        t0                = t,
                                        trading_frequency = trading_frequency,
                                        latency           = latency,
                                        initial_inventory = meta_order_size)        
                meta_order_id_c += 1

                verbose_print(verbose_level, t, trade_date, f'I received a meta order of size: {round(b_meta_order_arrival*meta_order_size)}. ID = {meta_order_id_c - 1} To be executed in T = {T} seconds.')

                # Sample from observed reward & Choose the right strategy
                # here is the TS stuff        
                best_strategy  = random.choice(list(strategies.keys()))

                verbose_print(verbose_level, t, trade_date, f'I choose the strategy: {best_strategy}')
                if len(rewards[best_strategy])>0:
                    verbose_print(verbose_level, t, trade_date, f'I have chosen the strategy: {best_strategy} with mean rewards : {round(np.mean(rewards[best_strategy]),2)}')


                o_strategy = strategy(strategy_config = strategies[best_strategy],
                                    strategy_id     = best_strategy,
                                    LOB_features    = LOB_features.iloc[:i].set_index("timestamp")[strategies[best_strategy]['params']['feature_name']])

                # link meta order to strategy object
                o_meta_order.strategy      = o_strategy
                o_meta_order.strategy_id   = o_strategy.strategy_id
                o_meta_order.t0            = t

                # tape the new meta order: replaced by pickling the meta order when the meta order is finished
                #  tape_meta_orders.append(o_meta_order)
                # add it to pending
                tape_pending_meta_orders[i] = o_meta_order

            ######################################
            # Execute all strategies 
            # Prepare orders for next time step
            # tape rewards if over
            ######################################
            for meta_order_item_k in tape_pending_meta_orders.keys():
                meta_order_item = tape_pending_meta_orders[meta_order_item_k]

                verbose_print(verbose_level, t, trade_date, f'Executing strategy for meta order = {meta_order_item.meta_order_id}.')
                o_order = meta_order_item.generate_strategy_order(order_id_c, t, feature_values=feature_values)

                if np.abs(o_order.quantity)>0:
                    verbose_print(verbose_level, t, trade_date, f'Adding strategic order to pending orders list. Meta order = {meta_order_item.meta_order_id}. Order = {o_order.order_id}. quantity = {o_order.quantity}')

                    tape_pending_orders.append(o_order)
                    order_id_c += 1
            
            # To keep track of time
            msg_ = 'Current: '+ str(round(i/len(times),4))+ "%, " + str(round(time.time() - start_time,2)) + " seconds"
            verbose_print(verbose_level, t, trade_date, msg_)



if __name__ == '__main__':
    strats = ['IMBALANCE', 'TREND']
    dynams = [1,2]

    paramlist = list(itertools.product(strats, dynams))

    N = mp.cpu_count()
    print('Number of parallelisable cores: ', N)

    with mp.Pool(processes = N) as p:
        p.map(sim, [paramlist[i] for i in range(len(paramlist))])
    # sim(['IMBALANCE', 1])
