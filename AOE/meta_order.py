from .order import order
import numpy as np
from collections import deque

class meta_order:
    def __init__(self,  **kwargs):
        self.meta_order_id     = kwargs['meta_order_id']
        self.quantity          = kwargs['quantity']            #if 'quantity'            in kwargs else 1
        self.S0                = kwargs['S0']                  #if 'S0'                  in kwargs else 0
        
        self.t0                = kwargs['t0']                  #if 't0'                  in kwargs else 0
        self.T                 = kwargs['T']                   #if 'T'                   in kwargs else 1000
        self.trading_frequency = kwargs['trading_frequency']   if 'trading_frequency'   in kwargs else 1
        self.latency           = kwargs['latency']             if 'latency'             in kwargs else 0
        
        self.initial_inventory = kwargs['initial_inventory']  #if 'initial_inventory' in kwargs else 0
        self.current_inventory = self.initial_inventory
        
        self.sent_orders_tape  = deque(maxlen=None)
        self.strategy          = None
        
        self.cash              = 0
        self.current_pnl       = 0
        self.last_trading_date = self.t0 # keeps track of last non zero order sent, for strategy purposes
    
    # This method is called when it is needed to update the inventory of the meta order, once an execution is confirmed
    def update_inventory(self, quantity, price):
        self.current_inventory += quantity # we suppose we are always executed
        self.cash              -= quantity*price
        self.current_pnl        = self.cash + self.current_inventory * price - self.initial_inventory * self.S0 # cash + qt St - q0 S0 
    
    
    # This method for generating orders from strategies. 
    # It calls the strategy object to provide the target inventory, and if needed, generated an order.
    def generate_strategy_order(self, order_id, t, **kwargs):
        # generate the order
        order_quantity = self.strategy.get_order_quantity(t                   = t, 
                                                          dt                  = self.trading_frequency , 
                                                          T                   = np.timedelta64(self.T, 's')+self.t0, 
                                                          t0                  = self.t0,
                                                          initial_inventory   = self.initial_inventory,
                                                          remaining_inventory = self.current_inventory,
                                                          last_trading_date   = self.last_trading_date,
                                                          **kwargs)

        o_order        = order(order_id       = order_id, 
                               quantity       = order_quantity, 
                               latency        = self.latency, 
                               the_meta_order = self) 
            
        if np.abs(order_quantity)>0:
            self.last_trading_date = t
            self.sent_orders_tape.append(o_order)

        return o_order
    
    # This method for generating orders with specified quantity
    def generate_order(self, order_id, order_quantity, t, **kwargs):
        # generate the order
        o_order        = order(order_id       = order_id, 
                               quantity       = order_quantity, 
                               latency        = self.latency, 
                               the_meta_order = self) 
            
        if np.abs(order_quantity)>0:
            self.sent_orders_tape.append(o_order)

        return o_order
    
    # This method tests if the meta order is finished, ONLY BASED ON TIME.
    # Raises a warning if the remaining inventory is not 0.
    def is_over(self, t):
        if t > self.t0 + self.T:
            if round(self.current_inventory) != 0: 
                return True, True 
            
            return True, False
        return False, False
