from AOE.utils import verbose_print
from AOE.optimal_trading import execute_strategy
from AOE.stats import get_meta_order_details
import scipy
import numpy as np

def update_reward_variables(order_arrival_time,  
                            historical_oracle_rewards, historical_oracle_strats, historical_all_rewards, historical_reward_times,
                            pending_rewards, verbose_level):
    to_pops = []
    for pending_reward_time in pending_rewards['oracle'].keys():
        if order_arrival_time >= pending_reward_time:

            historical_oracle_rewards  += [pending_rewards['oracle'][pending_reward_time]['oracle_reward']] #/10000
            historical_oracle_strats   += [pending_rewards['oracle'][pending_reward_time]['oracle_strat']]

            historical_all_rewards     += [[s for s in pending_rewards['oracle'][pending_reward_time]['all_strat_rewards']]] #/10000
            historical_reward_times    += [pending_rewards['oracle'][pending_reward_time]['order_arrival_time']]

            verbose_print(verbose_level, order_arrival_time, f'Adding an oracle reward obtained at {pending_reward_time}')
            verbose_print(verbose_level, order_arrival_time, f'Adding strats rewards obtained at {pending_reward_time}')
            
            to_pops  += [pending_reward_time]
            
    return to_pops


def update_bandit_variables(pending_int_rewards, pending_rewards, bandits, bandit_k, 
                            order_arrival_time,
                            verbose_level,
                            retrain_hyperparameters, historical_rewards, historical_strats,
                            nb_added_rewards):
    to_pops_int = []
    # Add intermediate rewards but do not tape them
    for pending_reward_time in pending_int_rewards[bandit_k].keys():
        if order_arrival_time >= pending_reward_time:
            verbose_print(verbose_level, order_arrival_time, f'Adding an intermediate reward for bandit {bandit_k} obtained at {pending_reward_time}', True)
            bandits[bandit_k].update_data(features  = pending_int_rewards[bandit_k][pending_reward_time]['features'], 
                                          strat     = pending_int_rewards[bandit_k][pending_reward_time]['strategy'], 
                                          reward    = pending_int_rewards[bandit_k][pending_reward_time]['reward'], #/10000 
                                          # reward_time = pending_int_rewards['TS'][pending_reward_time][5],
                                          retrain_hyperparameters = retrain_hyperparameters)
            
            nb_added_rewards[bandit_k]       += 1
            to_pops_int += [pending_reward_time]
    
    to_pops = []
    for pending_reward_time in pending_rewards[bandit_k].keys():
        if order_arrival_time >= pending_reward_time:
            verbose_print(verbose_level, order_arrival_time, f'Adding a final reward for bandit {bandit_k} obtained at {pending_reward_time}', True)

            bandits[bandit_k].update_data(features  = pending_rewards[bandit_k][pending_reward_time]['features'], 
                                          strat     = pending_rewards[bandit_k][pending_reward_time]['strategy'], 
                                          reward    = pending_rewards[bandit_k][pending_reward_time]['reward'], #/10000 
                                          # reward_time = pending_rewards['TS'][pending_reward_time][5],
                                          retrain_hyperparameters = retrain_hyperparameters)
            
            nb_added_rewards[bandit_k]       += 1
            historical_rewards[bandit_k]     += [pending_rewards[bandit_k][pending_reward_time]['reward']] #/10000
            historical_strats[bandit_k]      += [pending_rewards[bandit_k][pending_reward_time]['strategy']]
            to_pops                          += [pending_reward_time]
            
    return to_pops_int, to_pops 

def pop_from_dict(dict_to_change, to_pops):
    for to_pop in to_pops:
        dict_to_change.pop(to_pop)
        
# def get_strategies_rewards():
    
def execute_and_obtain_rewards(tape_meta_orders,
                               order_id_c, meta_order_id_c, strategies, LOB_features, 
                               order_arrival_time, T, trading_frequency,
                               meta_order_size, latency, trade_date, verbose_level, nb_intermediary_rewards):
    best_oracle_strategy  = list(strategies.keys())[0]
    best_oracle_reward    = None
    all_strats_rewards    = []

    order_terminal_time   = order_arrival_time + np.timedelta64(T, f's')
    trading_window_dates  = LOB_features.loc[order_arrival_time:order_terminal_time].index.values
    order_terminal_time   = trading_window_dates[-1]
    reward_info           = {}
    
    for strat in strategies.keys():
        order_id_c, meta_order_id_c, o_meta_order = execute_strategy(_strategy          = strategies[strat], 
                                                                     _trading_frequency = trading_frequency, 
                                                                     _t                 = order_arrival_time, 
                                                                     _T                 = T,
                                                                     _trading_times     = trading_window_dates,
                                                                     _meta_order_id_c   = meta_order_id_c,
                                                                     _order_id_c        = order_id_c,
                                                                     _meta_order_size   = meta_order_size,
                                                                     _latency           = latency, 
                                                                     _historical_feature_data = LOB_features)
        tape_meta_orders.append(o_meta_order)

        verbose_print(verbose_level, order_arrival_time, f'Executed strategy {strat} between {order_arrival_time} and {trading_window_dates[-1]}')

        initial_wealth      = o_meta_order.initial_inventory*o_meta_order.S0
        order_df            = get_meta_order_details(o_meta_order, trade_date)
        reward_indices      = [int(ii) for ii in np.linspace(0, len(order_df)-1, nb_intermediary_rewards+1)]
        int_reward_times    = order_df.iloc[reward_indices[1:-1]]['execution_time'].values
        int_rewards         = [100*rwd/initial_wealth for rwd in order_df.iloc[reward_indices[1:-1]]['meta_order_pnl'].values]

        final_strat_reward_time   = trading_window_dates[-1] #order_df.iloc[-1]['execution_time'] # final reward of the strategy
        final_strat_reward        = 100*order_df.iloc[-1]['meta_order_pnl']/initial_wealth # final reward of the strategy

        all_strats_rewards        += [final_strat_reward]

        reward_info[strat] = (int_reward_times, int_rewards, final_strat_reward_time, final_strat_reward) # record rewards

        verbose_print(verbose_level, order_arrival_time, f"Intermediary rewards: {[str(int_reward_times[i1])+ ':'+str(int_rewards[i1]) + '   ' for i1 in range(len(int_rewards)) ]}")
#        verbose_print(verbose_level, order_arrival_second, trade_date, f"Intermediary rewards times: {int_reward_times}")
        verbose_print(verbose_level, order_arrival_time, f"Final reward: {final_strat_reward}")

        if best_oracle_reward is None:
            best_oracle_reward   = final_strat_reward
            best_oracle_strategy = strat
        else:
            if final_strat_reward > best_oracle_reward:
                best_oracle_strategy = strat
                best_oracle_reward   = final_strat_reward
    
    
    return order_id_c, meta_order_id_c, reward_info, all_strats_rewards, best_oracle_reward, best_oracle_strategy

def update_pending_rewards(pending_int_rewards, pending_rewards, 
                           bandits, best_strategies_bandits, reward_info, 
                           feature_values, strategies,
                           best_oracle_strategy, best_oracle_reward, order_arrival_time, all_strats_rewards):
    
    for bandit_k in bandits.keys():
        selected_strategy = best_strategies_bandits[bandit_k]
        int_reward_times, int_rewards, final_strat_reward_time, final_strat_reward = reward_info[selected_strategy]

        # intermediate rewards
        for (i_rwd, int_reward_time) in enumerate(int_reward_times):
            pending_int_rewards[bandit_k][int_reward_time] = {'features'      : feature_values[[strategies[selected_strategy]['contextual_params']['feature_name']]], 
                                                              'strategy'      : selected_strategy, 
                                                              'reward'        : int_rewards[i_rwd]}

        # final reward
        pending_rewards[bandit_k][final_strat_reward_time] = {'features'      : feature_values[[strategies[selected_strategy]['contextual_params']['feature_name']]], 
                                                              'strategy'      : selected_strategy, 
                                                              'reward'        : final_strat_reward}

    # add oracle info
    pending_rewards["oracle"][final_strat_reward_time] = {'oracle_strat'       : best_oracle_strategy,
                                                          'oracle_reward'      : best_oracle_reward,
                                                          'order_arrival_time' : order_arrival_time,
                                                          'all_strat_rewards'  : all_strats_rewards}




                            
# class bandit_algo:
#     def __init__(self, algo, actions, N):
#         self.N = N
#         self.n = 0 # number of calls 
#         self.algorithm = algo # TS, UCB etc
#         self.actions = actions
#         self.n_selected = {action : 0 for action in self.actions}
#         self.Q = {action : np.random.uniform(0,3) for action in self.actions}
#         self.rewards = {action : [] for action in self.actions}

#     def update_reward(self, a, r):
#         self.rewards[a].append(r)

#     def select_arm(self):
#         self.n += 1
#         if self.algorithm == 'E-GREEDY':
#             return self.epsilon_greedy(epsilon=0.05)
#         elif self.algorithm == 'UCB1-NORMAL':
#             return self.UCB1_normal()
#         else:
#             print('ALGORITHM NOT IMPLEMENTED!')

#     def update_Q(self, r, a):
#         if self.algorithm == 'E-GREEDY':
#             curr_Q = self.Q[a]
#             self.Q[a] = curr_Q + (1/self.n_selected[a])*(r-curr_Q)

#     def epsilon_greedy(self, epsilon):
#         # e-greedy algorithm
#         if random.random() < epsilon:
#             a = random.choice(self.actions)
#         else:
#             a = max(self.Q, key=self.Q.get) # action with the highest mean reward

#         self.n_selected[a] += 1

#         return a

#     def UCB1_normal(self):
#         # UCB1-normal algorithm
        
#         max_UCI = 0
#         max_action = self.actions[0]
          
#         for a in self.actions:
#             # n>=2
#             if self.n_selected[a] <= 1:
#                 self.n_selected[a] += 1
#                 return a
#             # min ceiling
#             elif self.n_selected[a] < 8*np.log(self.n):
#                 self.n_selected[a] += 1
#                 return a

#             C = np.sqrt(16*np.var(self.rewards[a])*np.log(self.N-1)/self.n)
#             UCI = np.mean(self.rewards[a]) + C

#             if UCI >= max_UCI:
#                 max_action = a

#         self.n_selected[max_action] += 1
#         return max_action

#     def thompson_sampling(self):
#         print('')
