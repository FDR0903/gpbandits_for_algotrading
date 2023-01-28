class order:
    def __init__(self,  **kwargs):
        self.order_id        = kwargs['order_id']
        self.quantity        = kwargs['quantity']  if 'quantity'  in kwargs else 1
        self.latency         = kwargs['latency']   if 'latency'   in kwargs else 0       
        self.execution_price = 0
        self.execution_time  = 0
        self.the_meta_order  = kwargs['the_meta_order']   if 'the_meta_order'   in kwargs else None # pointer to meta_order
    
    def executed(self, execution_time, execution_price):
        # update the inventory of the meta_order
        self.the_meta_order.update_inventory(self.quantity, execution_price)
        
        # update the execution price of the order for later use
        self.execution_price = execution_price
        self.execution_time  = execution_time