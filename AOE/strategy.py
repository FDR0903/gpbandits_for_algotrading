from re import T
from scipy import interpolate
import numpy as np
import statsmodels.api as sm
import scipy.integrate as integrate

# This class handles the math of strategies
class strategy:
    def __init__(self, **kwargs):
        if 'strategy_config' in kwargs:
            self.strategy_config     = kwargs['strategy_config']
            self.strategy_name       = kwargs['strategy_config']['name']
            self.strategy_id         = kwargs['strategy_id']
            self.feature_name        = self.strategy_config['params']['feature_name']
            self.use_interpolator    = True

            if 'use_interpolator' in self.strategy_config['params']:
                self.use_interpolator    = self.strategy_config['params']['use_interpolator']
            
            LOB_features             = kwargs['LOB_features']
            # Additional strategy data for imbalance
            if self.strategy_name == 'imbalance':
                self.feature_data    = LOB_features.last(kwargs['strategy_config']['params']["estimation_period"])
                self.alpha           = kwargs['strategy_config']['params']["alpha"]
                self.phi             = kwargs['strategy_config']['params']["phi"]
                self.kappa           = kwargs['strategy_config']['params']["kappa"] # should be 1e-7 for 1h, 1e-8 for 6min and so on

                if self.use_interpolator:
                    self.interp_ts = kwargs['strategy_config']['interpolator'][1]
                    self.interp_qs = kwargs['strategy_config']['interpolator'][2]
                    self.interp_Is = kwargs['strategy_config']['interpolator'][3]
                    self.interp_thetaIs = kwargs['strategy_config']['interpolator'][4]
                    self.interp_obj = kwargs['strategy_config']['interpolator'][0]
            elif self.strategy_name == 'trend':
                self.feature_data    = LOB_features.last(kwargs['strategy_config']['params']["estimation_period"])
                self.alpha           = kwargs['strategy_config']['params']["alpha"]
                self.phi             = kwargs['strategy_config']['params']["phi"]
                self.kappa           = kwargs['strategy_config']['params']["kappa"] # should be 1e-7 for 1h, 1e-8 for 6min and so on
                
                if self.use_interpolator:
                    self.interp_ts = kwargs['strategy_config']['interpolator'][1]
                    self.interp_qs = kwargs['strategy_config']['interpolator'][2]
                    self.interp_Is = kwargs['strategy_config']['interpolator'][3]
                    self.interp_thetaIs = kwargs['strategy_config']['interpolator'][4]
                    self.interp_obj = kwargs['strategy_config']['interpolator'][0]
        else:
            self.strategy_name       = kwargs['strategy_name']      #if 'strategy_name'  in kwargs else 0    

        if ((self.strategy_name == 'imbalance') | (self.strategy_name == 'trend')):
            self.first_time_called = True
            
    def get_order_quantity(self, **kwargs):
        if self.strategy_name == 'TWAP':
            return self.TWAP(**kwargs)
        
        if self.strategy_name == 'imbalance':
            return self.IMBALANCE(**kwargs)
        
        if self.strategy_name == 'trend':
            return self.TREND(**kwargs)
            
        if self.strategy_name == 'depth':
            return self.depth(**kwargs)
    
    # Gives the target inventory for a TWAP strategy
    def TWAP(self, **kwargs):
        t0, t, T, initial_inventory, remaining_inventory = kwargs['t0'], kwargs['t'], kwargs['T'], kwargs['initial_inventory'], kwargs['remaining_inventory']

        # transform dates in seconds
        T_seconds = (T - t0)/np.timedelta64(1, f's')
        t_seconds = (t - t0)/np.timedelta64(1, f's')

        # linear interpolation
        y_interp = interpolate.interp1d([0, T_seconds], [initial_inventory, 0])
        try:
            return round(y_interp(t_seconds) - remaining_inventory)
        except Exception as e:
            print(str(e))
            print(t0, t, T, initial_inventory, remaining_inventory)
            raise Exception("Error while computing quantity for TWAP")
    
    


#     #######################################
#     # TREND
#     #######################################
#     def TREND_estimate_ou_params(self, feature_data, dn): # dt is in trade time
#         y   = feature_data.shift(-dn).iloc[:-dn].values
#         x   = feature_data.iloc[:-dn].values
#         model   = sm.OLS(y, x)
        # results = model.fit()
        # r       = (1-results.params[0])/dn
        # sigmaI  = results.resid.std() / dn**0.5
        
#         return r, sigmaI, results.rsquared
    
#     def TREND_getOptimalSpeed(self, qt, It, t, T, phi, kappa, alpha, thetaI, sigmaI):
# #         print('Calling optimalspeed with inventory=', qt, 'and It=', It, 'at t =', t/T)
#         if self.use_interpolator:
#             if ((t > 0.95 * T) |
#                     (t > self.interp_ts[-2]) | (t < self.interp_ts[0]) | 
#                     (qt > self.interp_qs[-1]) | (qt < self.interp_qs[0]) |
#                     (It > self.interp_Is[-1]) | (It < self.interp_Is[0])  | 
#                     (thetaI > self.interp_thetaIs[-1]) | (thetaI < self.interp_thetaIs[0])):
#                 print((t, qt, It, thetaI))
#                 print('*******************************')
#                 print('MEEEEEEEEEEEEEEEEEEEEERDE')
#                 print('*******************************')
#                 int_v2t = lambda s: integrate.quad(lambda t:self.v2(t, phi, kappa, alpha, T), t, s)[0]
#                 to_integrate = lambda s: np.exp(-thetaI * (s-t) + int_v2t(s) /kappa)
#                 optimalspeed = integrate.quad(to_integrate, t, T)[0]
#                 return -(optimalspeed*It + 2*self.v2(t, phi, kappa, alpha, T)*qt)/(2*kappa) 
#             else:
#                 return self.interp_obj((t, qt, It, thetaI))
#         else:
#             int_v2t = lambda s: integrate.quad(lambda t:self.v2(t, phi, kappa, alpha, T), t, s)[0]
#             to_integrate = lambda s: np.exp(-thetaI * (s-t) + int_v2t(s) /kappa)
#             optimalspeed = integrate.quad(to_integrate, t, T)[0]
#             return -(optimalspeed*It + 2*self.v2(t, phi, kappa, alpha, T)*qt)/(2*kappa) 
            
    
    def v2(self, t, phi, kappa, alpha, T):
        beta = np.sqrt(phi/kappa)
        xi   = (alpha + np.sqrt(kappa*phi)) / (alpha - np.sqrt(kappa*phi))
        return np.sqrt(kappa*phi) * (1 + xi*np.exp(2*beta*(T-t))) / (1 - xi*np.exp(2*beta*(T-t)))

#     # needs feature_data, alpha, phi, kappa, It
#     def TREND(self, **kwargs):
#         # initialize what needs to be
#         remaining_inventory = kwargs['remaining_inventory']
#         t                   = kwargs['t']

#         if self.first_time_called:
#             self.first_time_called = False
#             self.t0, self.T, self.initial_inventory = kwargs['t0'], kwargs['T'], kwargs['initial_inventory']            
            
#             # everything is in number of seconds, transform
#             self.T = (self.T - self.t0)/(60*60*24)
            
#             # calibrate parameters
#             self.thetaI, self.sigmaI, _ = self.TREND_estimate_ou_params(self.feature_data, dn=5)

#             self.feature_data = None # for memory use
#         # otherwise just give the target inventory
#         dt     = (t - kwargs['last_trading_date'])/(60*60*24)
#         t      = (t - self.t0)/(60*60*24)
        

#         optimal_speed = dt * self.TREND_getOptimalSpeed(qt  = remaining_inventory, 
#                                                       It  = kwargs['feature_values'][self.feature_name ], 
#                                                       t   = t, 
#                                                       T   = self.T, 
#                                                       phi = self.phi, kappa = self.kappa, alpha = self.alpha, thetaI = self.thetaI, sigmaI=self.sigmaI)
#         return -round(optimal_speed)


    #######################################
    # IMBALANCE
    #######################################
    def IMB_estimate_ou_params(self, feature_data, dn): # dt is in trade time
        y   = feature_data.shift(-dn).iloc[:-dn].values
        x   = feature_data.iloc[:-dn].values
        model   = sm.OLS(y, x)
        results = model.fit()
        r      = (1-results.params[0])/dn
        sigmaI = results.resid.std() / dn**0.5

        return r, sigmaI, results.rsquared

    def IMB_getOptimalSpeed(self, qt, It, t, T, phi, kappa, alpha, thetaI, sigmaI):
#         print('Calling optimalspeed with inventory=', qt, 'and It=', It, 'at t =', t/T)
        # test if interpolator works
        if self.use_interpolator:
            if ((t > 0.95 * T) |
                    (t > self.interp_ts[-2]) | (t < self.interp_ts[0]) | 
                    (qt > self.interp_qs[-1]) | (qt < self.interp_qs[0]) |
                    (It > self.interp_Is[-1]) | (It < self.interp_Is[0])  | 
                    (thetaI > self.interp_thetaIs[-1]) | (thetaI < self.interp_thetaIs[0])):
                print((t, qt, It, thetaI))
                print('*******************************')
                print('MEEEEEEEEEEEEEEEEEEEEERDE')
                print('*******************************')
                int_v2t = lambda s: integrate.quad(lambda t:self.v2(t, phi, kappa, alpha, T), t, s)[0]
                to_integrate = lambda s: np.exp(-thetaI * (s-t) + int_v2t(s) /kappa)
                optimalspeed = integrate.quad(to_integrate, t, T)[0]
                return -(optimalspeed*It + 2*self.v2(t, phi, kappa, alpha, T)*qt)/(2*kappa) 
            else:
                return self.interp_obj((t, qt, It, thetaI))
        else:
            int_v2t = lambda s: integrate.quad(lambda t:self.v2(t, phi, kappa, alpha, T), t, s)[0]
            to_integrate = lambda s: np.exp(-thetaI * (s-t) + int_v2t(s) /kappa)
            optimalspeed = integrate.quad(to_integrate, t, T)[0]
            return -(optimalspeed*It + 2*self.v2(t, phi, kappa, alpha, T)*qt)/(2*kappa) 
    
    # needs feature_data, alpha, phi, kappa, It
    def IMBALANCE(self, **kwargs):
        # initialize what needs to be
        remaining_inventory = kwargs['remaining_inventory']
        t                   = kwargs['t']

        if self.first_time_called:
            self.first_time_called = False
            self.t0, self.T, self.initial_inventory = kwargs['t0'], kwargs['T'], kwargs['initial_inventory']            
            
            # transform dates in seconds
            self.T_seconds = (self.T - self.t0) / np.timedelta64(1, 's')

            # everything is in number of seconds, transform
            self.T = self.T_seconds/(60*60*24) # (self.T - self.t0)/(60*60*24)
            
            # calibrate parameters
            self.thetaI, self.sigmaI, _ = self.IMB_estimate_ou_params(self.feature_data, dn=5)

            self.feature_data = None # for memory

        # otherwise just give the target inventory
        last_trading_date_seconds = (kwargs['last_trading_date'] - self.t0) / np.timedelta64(1, 's')

        t_seconds  = (t - self.t0) / np.timedelta64(1, 's')
        dt         = (t_seconds - last_trading_date_seconds)/(60*60*24)

        
        optimal_speed = dt * self.IMB_getOptimalSpeed(qt  = remaining_inventory, 
                                                      It  = kwargs['feature_values'][self.feature_name], 
                                                      t   = t_seconds/(60*60*24), 
                                                      T   = self.T, 
                                                      phi = self.phi, kappa = self.kappa, alpha = self.alpha, thetaI = self.thetaI, sigmaI=self.sigmaI)
        return -round(optimal_speed)
    
    

