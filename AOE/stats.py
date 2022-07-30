import datetime
import pandas as pd
import numpy as np
from scipy import interpolate

def get_meta_order_df(tape_meta_orders, trade_date):
    tape_meta_orders_ids        = [tape_meta_orders[j].meta_order_id for j in range(len(tape_meta_orders))]
    tape_meta_orders_quantities = [tape_meta_orders[j].quantity for j in range(len(tape_meta_orders))]
    tape_meta_orders_S0s        = [tape_meta_orders[j].S0 for j in range(len(tape_meta_orders))]

    tape_meta_orders_t0s        = [datetime.datetime.strptime(trade_date, '%Y-%m-%d') + \
                                    datetime.timedelta(seconds=tape_meta_orders[j].t0) \
                                    for j in range(len(tape_meta_orders))]

    tape_meta_orders_Ts         = [tape_meta_orders[j].T for j in range(len(tape_meta_orders))]

    tape_meta_orders_final_inventories          = [tape_meta_orders[j].current_inventory for j in range(len(tape_meta_orders))]
    tape_meta_orders_ordernumber                = [len(tape_meta_orders[j].sent_orders_tape) for j in range(len(tape_meta_orders))]
    tape_meta_orders_strategies                 = [tape_meta_orders[j].strategy_id for j in range(len(tape_meta_orders))]
    tape_meta_orders_finacashes                 = [tape_meta_orders[j].cash for j in range(len(tape_meta_orders))]
    tape_meta_orders_pnls                       = [tape_meta_orders[j].current_pnl for j in range(len(tape_meta_orders))]
    
    tape_meta_orders_last_order_times           = []
    
    for j in range(len(tape_meta_orders)):
        if len(tape_meta_orders[j].sent_orders_tape)>0:
            tape_meta_orders_last_order_times           += [datetime.datetime.strptime(trade_date, '%Y-%m-%d') + \
                                                datetime.timedelta(seconds=tape_meta_orders[j].sent_orders_tape[-1].execution_time)]
        else:
            tape_meta_orders_last_order_times           = tape_meta_orders_t0s[j]
        

    meta_order_df = pd.DataFrame.from_dict({'id'                   : tape_meta_orders_ids,
                                            'strategy_name'        : tape_meta_orders_strategies,
                                            'quantity'             : tape_meta_orders_quantities,
                                            'initial_price'        : tape_meta_orders_S0s,
                                            'initial_time'         : tape_meta_orders_t0s,
                                            'final_execution_time' : tape_meta_orders_last_order_times,
                                            'final_inventory'      : tape_meta_orders_final_inventories,
                                            'execution_window'     : tape_meta_orders_Ts,
                                            'orders_number'        : tape_meta_orders_ordernumber,
                                            'final_cash'           : tape_meta_orders_finacashes,
                                            'final_pnl'            : tape_meta_orders_pnls                                       
                                        }).set_index('id')
    return meta_order_df

def get_meta_order_details(o_meta_order, trade_date):
    o_meta_order_child_orders    = o_meta_order.sent_orders_tape

    meta_order_initial_inventory = o_meta_order.initial_inventory
    meta_order_initial_price     = o_meta_order.S0

    meta_order_order_ids         = meta_order_initial_inventory + np.cumsum([o_meta_order_child_orders[j].order_id for j in range(len(o_meta_order_child_orders))])
    meta_order_inventory         = meta_order_initial_inventory + np.cumsum([o_meta_order_child_orders[j].quantity for j in range(len(o_meta_order_child_orders))])
    meta_order_execution_prices  = np.array([o_meta_order_child_orders[j].execution_price for j in range(len(o_meta_order_child_orders))])
    meta_order_execution_qtties  = np.array([o_meta_order_child_orders[j].quantity for j in range(len(o_meta_order_child_orders))])
    
    meta_order_execution_times   = [o_meta_order_child_orders[j].execution_time for j in range(len(o_meta_order_child_orders))]
    meta_order_execution_seconds = [ (o_meta_order_child_orders[j].execution_time - o_meta_order_child_orders[0].execution_time) / np.timedelta64(1, 's')  for j in range(len(o_meta_order_child_orders)) ]
    # meta_order_execution_times   = [datetime.datetime.strptime(trade_date, '%Y-%m-%d') + \
    #                                 datetime.timedelta(seconds=o_meta_order_child_orders[j].execution_time) \
    #                                 for j in range(len(o_meta_order_child_orders))]
    meta_order_cash_process      = -np.cumsum(meta_order_execution_prices * meta_order_execution_qtties)
    meta_order_pnl_process       = meta_order_cash_process + meta_order_inventory*meta_order_execution_prices - \
                                    meta_order_initial_inventory * meta_order_initial_price

    order_df      = pd.DataFrame.from_dict({'order_id'             : meta_order_order_ids,
                                            'quantity'             : meta_order_execution_qtties,
                                            'execution_price'      : meta_order_execution_prices,
                                            'execution_time'       : meta_order_execution_times,
                                            
                                            'meta_order_inventory' : meta_order_inventory,
                                            'meta_order_cash'      : meta_order_cash_process,
                                            'meta_order_pnl'       : meta_order_pnl_process                                  
                                        }).set_index('order_id')
    
    # TWAP information
    y_interp = interpolate.interp1d([0, meta_order_execution_seconds[-1]], [meta_order_initial_inventory, 0.])
    
    
    order_df['TWAP_inventory']    = [ round(y_interp(float(meta_order_execution_seconds[i]))+0) for i in  range(len(order_df.execution_time.values))]
    order_df['TWAP_quantity']     = order_df['TWAP_inventory'].diff(1).fillna(0)
    
    order_df['TWAP_cash']         = -np.cumsum(order_df['execution_price'] * order_df['TWAP_quantity'])
    order_df['TWAP_pnl']          = order_df['TWAP_cash'] + order_df['TWAP_inventory'] * order_df['execution_price'] - \
                                        o_meta_order.initial_inventory * o_meta_order.S0    

    return order_df