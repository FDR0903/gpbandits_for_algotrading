import numpy as np

from AOE.order import order
from AOE.meta_order import meta_order
from AOE.strategy import strategy
from AOE.utils import verbose_print



# Takes relevant data, strategy configuration, and returns a meta_order object along with rewards and reward times
def execute_strategy(_strategy, 
                     _trading_frequency, 
                     _t,                      # Current time
                     _T,                      # Expiry of the meta order, in seconds
                     _trading_times,          # = [_t, ...., _t + _T]
                     _meta_order_id_c, 
                     _order_id_c,
                     _meta_order_size,
                     _latency, 
                     _historical_feature_data ,# For estimation of the signal OU dynamics at the beginning of the execution        
                     _tick_size
                    ):
    strategy_name = _strategy['name']
    _current_price = _historical_feature_data.loc[_t, 'mid_price']
    
    # create the meta order object
    o_meta_order  = meta_order(meta_order_id      = _meta_order_id_c,             
                                S0                = _current_price,
                                quantity          = _meta_order_size, 
                                T                 = _T, #trading_times[-1] - _t,
                                t0                = _t,
                                trading_frequency = _trading_frequency,
                                latency           = _latency,
                                initial_inventory = _meta_order_size)       
    
    # create the strategy object    
    o_strategy = strategy(strategy_config = _strategy,
                          strategy_id     = strategy_name,
                          LOB_features    = _historical_feature_data.loc[:_t][_strategy['params']['feature_name']])
    
    # link meta order to strategy object
    o_meta_order.strategy      = o_strategy
    o_meta_order.strategy_id   = o_strategy.strategy_id
    o_meta_order.t0            = _t
    
    # Loop through the trading window
    last_trading_time = _t
    
    print('execution and generating orders')
    for (t_1, t) in zip(_trading_times, _trading_times[1:]):
        # only trade if trading frequency allows
        if (t - last_trading_time)/np.timedelta64(1, 's') > _trading_frequency:
            last_trading_time =  t

            # Get market information
            feature_values = _historical_feature_data.loc[t_1, :]
            St             = _historical_feature_data.loc[t, 'mid_price']
            bestbid        = St - _tick_size
            bestask        = St + _tick_size
            
            # Execute strategy algo to get the order      
            o_order = o_meta_order.generate_strategy_order(_order_id_c, t, 
                                                            feature_values = feature_values)

            if np.abs(o_order.quantity)>0:
                # Execute at the future price
                if o_order.quantity<0: # hit the best bid
                    o_order.executed(t, bestbid)
                else: #hit the best ask
                    o_order.executed(t, bestask)
                    
                _order_id_c += 1
                
            
            
    _meta_order_id_c += 1
    
    return _order_id_c, _meta_order_id_c, o_meta_order