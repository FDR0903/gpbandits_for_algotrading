import pandas as pd, numpy as np
from IPython.display import clear_output
import datetime
import os
import pathlib
import math

def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()


def rsi(close, n=14, rtrfreq=1, fillna=False): # n in number of trades
    diff = close.diff(rtrfreq)
    which_dn = diff < 0

    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = ema(up, n, fillna)
    emadn = ema(dn, n, fillna)

    rsi = 100 * emaup / (emaup + emadn)
    if fillna:
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(rsi, name='rsi')

def get_LOB_data(data_path, asset_name, trade_date):
    # prices are in: USD * 10000
    try:
        LOB_data_path = os.path.join(data_path, asset_name, f'{asset_name}_{trade_date}_34200000_57600000_orderbook_5.csv')
        LOB_data      = pd.read_csv(LOB_data_path,
                                    engine    = 'c',
                                    index_col = None,
                                    header    = None,
                                    infer_datetime_format=True)
        
        
    except:
        LOB_data_path1 = os.path.join(data_path, asset_name, f'{asset_name}_{trade_date}_34200000_57600000_orderbook_5_part1.csv')
        LOB_data_path2 = os.path.join(data_path, asset_name, f'{asset_name}_{trade_date}_34200000_57600000_orderbook_5_part2.csv')
        LOB_data1      = pd.read_csv(LOB_data_path1,
                                    engine    = 'c',
                                    index_col = 0,
                                    header    = None,
                                    infer_datetime_format=True)
        LOB_data2      = pd.read_csv(LOB_data_path2,
                                    engine    = 'c',
                                    index_col = 0,
                                    header    = None,
                                    infer_datetime_format=True)
        LOB_data       = pd.concat((LOB_data1, LOB_data2.iloc[1:,:]))

    LOB_messages      = pd.read_csv(f'{data_path}/{asset_name}/{asset_name}_{trade_date}_34200000_57600000_message_5.csv', 
                                    engine    = 'c',
                                    index_col = 0,
                                    header    = None)
     
    LOB_data.index = LOB_messages.index
    LOB_data_columns = []
    for i in range(len(LOB_data.columns)//4): 
        LOB_data_columns += [f'ask_{i+1}', f'ask_volume_{i+1}', f'bid_{i+1}', f'bid_volume_{i+1}']

    LOB_data.columns = LOB_data_columns
    LOB_data.index.name = 'time'
    LOB_data = LOB_data.reset_index(drop=False).groupby('time').last()
    
    LOB_messages.drop(LOB_messages.columns[len(LOB_messages.columns)-1], axis=1, inplace=True)
    LOB_messages.columns = ['Event Type', 'Order ID', 'Size', 'Price', 'Direction']

    return LOB_data, LOB_messages


def get_LOB_features(LOB_data, trade_date, **kwargs):
    LOB_msg        = kwargs['LOB_msg']
    _option        = False
    if 'option' in kwargs: _option        = kwargs['option']
    
    LOB_data['mid_price']   = (LOB_data['ask_1'] + LOB_data['bid_1'])/2
    LOB_features            = LOB_data[['bid_1', 'bid_volume_1', 'mid_price', 'ask_1', 'ask_volume_1']]
    
    # Imbalances
    LOB_features['imbalance_1'] = (LOB_data['bid_volume_1'] - LOB_data['ask_volume_1'])/(LOB_data['bid_volume_1'] + LOB_data['ask_volume_1'])
    LOB_features['imbalance_2'] = (LOB_data[['bid_volume_1', 'bid_volume_2']].sum(axis=1) - LOB_data[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1))/ \
                                   (LOB_data[['bid_volume_1', 'bid_volume_2']].sum(axis=1) + LOB_data[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1))
    LOB_features['imbalance_3'] = (LOB_data[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1) - LOB_data[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1))/ \
                                   (LOB_data[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1) + LOB_data[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1))
    
    # Time given trade date
    LOB_features['timestamp'] = [datetime.datetime.strptime(trade_date, '%Y-%m-%d') + datetime.timedelta(seconds=i) for i in LOB_features.index]    
    LOB_features['traded_volume'] = LOB_data[[s for s in LOB_data.columns if 'volume' in s]].diff(1).fillna(0).abs().sum(axis=1)

    # Order book depth
    # LOB_features['bid_depth'] = LOB_data[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1).rolling(depth_config['w']).mean()
    # LOB_features['ask_depth'] = LOB_data[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1).rolling(depth_config['w']).mean()
    
    # Fwd price moves
    for w in (1, 5, 10, 20, 50, 40, 60, 100, 300, 500, 1000, 5000, 10000, 20000, 50000, 100000):
        LOB_features[f'fwd_price_move_{w}'] = (LOB_features['mid_price']/10000).diff(w).shift(-w)

    # Tred, volatility, RSI, and Liquidity Taking Index
    for w in (1000, 10000, 50000, 100000):
        LOB_features[f'trend_{w}']         = (LOB_features.mid_price - LOB_features.mid_price.ewm(halflife=w).mean())/10000

    for (w, w1) in zip((1000, 10000, 50000, 100000), (1, 10, 50, 100)):
        LOB_features[f'vol_{w1}_{w}']         = (LOB_features['mid_price']/10000).fillna(method='ffill').fillna(method='bfill').pct_change(w1).rolling(w).std() 
    
    for (w, w1) in zip((10000, 50000, 100000), (5, 25, 50)):
        LOB_features[f'rsi_{w1}_{w}'] = rsi(LOB_features.mid_price/10000, w, w1)

    # traded quantity
    askchanges = LOB_data[['ask_1']].diff(1)
    bidchanges = LOB_data[['bid_1']].diff(1)
    askchanges = askchanges[askchanges!=0].dropna()
    bidchanges = bidchanges[bidchanges!=0].dropna()
    tradedqtities = LOB_data[[s for s in LOB_data.columns if 'volume_1' in s]].diff(1)
    tradedqtities.loc[askchanges.index, 'ask_volume_1'] = 0
    tradedqtities.loc[bidchanges.index, 'bid_volume_1'] = 0

    # bid & ask depletions
    tradedqtities.loc[askchanges[askchanges>0].dropna().index, 'ask_volume_1'] = -LOB_features.shift(1).loc[askchanges[askchanges>0].dropna().index, 'ask_volume_1']
    tradedqtities.loc[bidchanges[bidchanges<0].dropna().index, 'bid_volume_1'] = -LOB_features.shift(1).loc[bidchanges[bidchanges<0].dropna().index, 'bid_volume_1']
    tradedqtities.loc[tradedqtities[tradedqtities.ask_volume_1>0].index, 'ask_volume_1'] = 0
    tradedqtities.loc[tradedqtities[tradedqtities.bid_volume_1>0].index, 'bid_volume_1'] = 0
    
    LOB_features['traded_ask_qtties'] = tradedqtities['ask_volume_1'].abs().values
    LOB_features['traded_bid_qtties'] = tradedqtities['bid_volume_1'].abs().values
    LOB_features['traded_qtties'] = LOB_features['traded_ask_qtties']  + LOB_features['traded_bid_qtties']

    # BA spreads
    LOB_features['ba_spread'] = (LOB_data['ask_1']  - LOB_data['bid_1'])/10000
    LOB_features['ba_spread_1'] = LOB_features['ba_spread'].ewm(halflife=10).mean().round(3)
    LOB_features['ba_spread_1'] = LOB_features['ba_spread'].ewm(halflife=20).mean().round(3)
    LOB_features['ba_spread_2'] = LOB_features['ba_spread'].ewm(halflife=50).mean().round(3)
    LOB_features['ba_spread_3'] = LOB_features['ba_spread'].ewm(halflife=100).mean().round(3)
    LOB_features['ba_spread_4'] = LOB_features['ba_spread'].ewm(halflife=500).mean().round(3)

    LOB_features['bid_1']     = LOB_features['bid_1'] / 10000
    LOB_features['mid_price'] = LOB_features['mid_price'] / 10000
    LOB_features['ask_1']     = LOB_features['ask_1'] / 10000

    # add some other contextual features by the second
    if _option:
        ctxt_features    = pd.DataFrame(columns=['newLO', 'newCancel', 'newMO', 'Intensity'])
        start_sec        = int(LOB_data.index.values[0])
        time_last_trade  = 0
        shares_added     = 0
        shares_executed  = 0
        shares_cancelled = 0
        X = 0
        tc = 1
        for t in range(len(LOB_msg)):
            # Classification based on event type
            event_type = LOB_msg.iloc[t]['Event Type']

            if event_type == 1: # New LO
                shares_added += LOB_msg.iloc[t]['Size'] # Running sum volume
            elif event_type == 2: # Partial Cancellation
                shares_cancelled += LOB_msg.iloc[t]['Size'] # Running sum cancelled orders
            elif event_type == 3: # Total Cancellation
                shares_cancelled += LOB_msg.iloc[t]['Size'] # Running sum cancelled orders
            elif event_type == 4: # Executed visible MO
                St = LOB_msg.iloc[t]['Size']
                shares_executed += St # Running sum executed orders
                delta_t = LOB_msg.index.values[t] - time_last_trade 
                X = X * math.exp(-delta_t/tc) + St
                # Next iteration
                time_last_trade = LOB_msg.index.values[t]
            elif event_type == 5: # Executed hidden MO
                St = LOB_msg.iloc[t]['Size']
                shares_executed += St # Running sum executed orders
                delta_t = LOB_msg.index.values[t] - time_last_trade 
                X = X * math.exp(-delta_t/tc) + St
                # Next iteration
                time_last_trade = LOB_msg.index.values[t]
                
            if int(LOB_msg.index.values[t]) > start_sec: # if second has passed
                start_sec = int(LOB_msg.index.values[t])
                feature_vector = [shares_added,  shares_cancelled, shares_executed, X]
                ctxt_features.loc[LOB_msg.index.values[t]] = feature_vector
                # Make all running sums zero
                shares_added = 0
                shares_executed = 0
                shares_cancelled = 0
        
        
        for w in (100, 500, 1000, 10000):
            LOB_features[f'LT_{w}'] = (np.sign(LOB_features['mid_price'].diff(w))*(LOB_features.newMO.rolling(w).mean() / (LOB_features.newMO.rolling(w).mean() + LOB_features.newLO.rolling(w).mean())))

        
        return pd.concat((LOB_features, 
                        ctxt_features[['newLO', 'newCancel', 'newMO', 'Intensity']]), axis=1
                        ).sort_index().fillna(method='ffill')
    else:
        return LOB_features.sort_index().fillna(method='ffill')


def get_meta_order_df(reward_path, asset_name, trade_date):
    meta_order_df = pd.read_csv(pathlib.Path(reward_path,
                                             f"meta_order_df_{asset_name}_{trade_date}.csv"))
    meta_order_df["initial_time"] = pd.to_datetime(meta_order_df["initial_time"])
    meta_order_df["final_execution_time"] = pd.to_datetime(meta_order_df["final_execution_time"])    
    meta_order_df = meta_order_df[-meta_order_df.final_inventory.astype(float).abs()>-10]    
    meta_order_df = meta_order_df.set_index(['id', 'strategy_name'])
    meta_order_df = meta_order_df[-meta_order_df.final_inventory.astype(float).abs()>-10]
    meta_order_df['final_pnl_twap'] = meta_order_df['final_pnl'] - meta_order_df['twap_pnl']
    return meta_order_df

def get_features(LOB_data, trade_date, **kwargs):
    LOB_data['mid_price']   = (LOB_data['ask_1'] + LOB_data['bid_1'])/2
    LOB_features            = LOB_data[['bid_1', 'bid_volume_1', 'mid_price', 'ask_1', 'ask_volume_1']]
    
    LOB_features['imbalance_1'] = (LOB_data['bid_volume_1'] - LOB_data['ask_volume_1'])/(LOB_data['bid_volume_1'] + LOB_data['ask_volume_1'])
    LOB_features['imbalance_2'] = (LOB_data[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1) - LOB_data[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1))/ \
                                   (LOB_data[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1) + LOB_data[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1))
    
    trend_configs  = kwargs['trend']
    depth_config  = kwargs['depth']
    
    LOB_features['bid_depth'] = LOB_data[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1).rolling(depth_config['w']).mean()
    LOB_features['ask_depth'] = LOB_data[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1).rolling(depth_config['w']).mean()
    
    LOB_features['timestamp'] = [datetime.datetime.strptime(trade_date, '%Y-%m-%d') + datetime.timedelta(seconds=i) for i in LOB_features.index]    
    LOB_features['traded_volume'] = LOB_data[[s for s in LOB_data.columns if 'volume' in s]].diff(1).fillna(0).abs().sum(axis=1)
    
    
    for (i, trend_config) in enumerate(trend_configs):
        trend_indicator =  (LOB_features.set_index('timestamp').mid_price - \
                             LOB_features.set_index('timestamp').mid_price.rolling(trend_config['w']).mean())
        trend_indicator_rolling_max  = trend_indicator.abs().rolling(trend_config['w']).max()        
        LOB_features[f'trend_{i}'] = (trend_indicator/trend_indicator_rolling_max).fillna(0).values

    return LOB_features




def verbose_print(verbose_level, t, msg, mode_=False):
    if verbose_level:
#        real_time = datetime.datetime.strptime(trade_date, '%Y-%m-%d') + datetime.timedelta(seconds=t)
        if mode_: print('\n******* ', t, " *******")
        print(msg)
#        clear_output(wait=True)