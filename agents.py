import random
import networkx as nx

class Order:
    """Links restaurants, users and drivers"""
    def __init__(self, order_id, user_id, restaurant_id, prep_time, start_time,route_to_user):
        self.id = order_id
        self.user_id = user_id
        self.restaurant_id = restaurant_id
        self.driver_id = None
        
        # Timing attributes
        self.prep_time = prep_time
        self.start_time = start_time
        self.ready_time = start_time + prep_time
        
        # Statuses: 'PREPARING', 'READY', 'PICKED_UP', 'DELIVERED'
        self.status = 'PREPARING'
        self.route_to_user = route_to_user 

class Restaurant:
    """Fullfils orders"""
    def __init__(self, restaurant_id: int, location: int, rating: float, 
                 capacity: int, avg_prep_time: float,service_radius: float, 
                 enabled=True):
        self.id = restaurant_id
        self.location = location  # node id in graph
        self.rating = rating 
        self.capacity = capacity
        self.avg_prep_time = avg_prep_time # seconds 
        self.service_radius = service_radius # Maximum delivery distance in meters
        self.active_orders = []  # Holds orders currently in preparation or ready
        self.enabled = enabled

    def can_accept_order(self):
        """Validates if the restaurant can take more orders based on capacity and status"""
        if not self.enabled:
            return False
        if len(self.active_orders) >= self.capacity:
            return False
        # 1% random rejection probability to simulate real-world glitches
        return random.random() > 0.01 

    def update_preparing_orders_to_ready(self, current_time):
        """Transitions orders from PREPARING to READY if prep time has elapsed"""
        for order in self.active_orders:
            if order.status == 'PREPARING' and current_time >= order.ready_time:
                order.status = 'READY'

    def generate_prep_time(self) -> float:
        """Calculates prep time using an exponential distribution"""
        return random.expovariate(1.0 / self.avg_prep_time)
    
    def accept_order(self, order: Order):
        """Adds a new order instance to the active queue"""
        self.active_orders.append(order)
        self._sync_enabled_status()

    def remove_order(self, order: Order):
            """
            Call this when a driver picks up the order.
            This is the ONLY moment capacity is actually freed.
            """
            if order in self.active_orders:
                self.active_orders.remove(order)
                self._sync_enabled_status()

    def _sync_enabled_status(self):
            """
            Automatically toggles the restaurant status based on current capacity.
            This acts as a safety 'circuit breaker'.
            """
            if len(self.active_orders) >= self.capacity:
                self.enabled = False
            else:
                # You might want to keep it False if it was manually disabled, 
                # but for this auto-scaling logic, we re-enable it when space opens up.
                self.enabled = True

class User:
    def __init__(self,user_id :int,location: int):
        self.user_id = user_id
        self.location = location

class Simulation:
    def __init__(self,graph, step_size=1):
        self.current_time = 0       # Total seconds elapsed
        self.step_size = step_size  # Duration of each tick
        self.restaurants: dict[int,Restaurant] = {} # Dictionary for O(1) restaurant lookup
        self.users : dict[int,User] = {} # Dictionary for O(1) restaurant lookup
        self.drivers = []           # Placeholder for future Driver agents
        self.orders = []            # Global order history for analytics
        self.order_id_counter = 1
        self.route_cache = {}
        self.graph = graph

    def add_restaurant(self, restaurant: Restaurant):
        self.restaurants[restaurant.id] = restaurant

    def add_user(self,user: User):
        self.users[user.user_id] = user

    def get_route_data(self, origin_node, destination_node):
        """
        Retrieves routing data. If it's the first time these nodes are connected,
        it calculates the path and caches it.
        """
        if origin_node == destination_node:
            return 0, [origin_node]
            
        # We use a tuple of nodes as the key. This allows different users 
        # at the same intersection to share the same path data.
        cache_key = (origin_node, destination_node)
        
        if cache_key not in self.route_cache:
            try:
                # Use the graph passed during Simulation initialization
                distance = nx.shortest_path_length(self.graph, origin_node, destination_node, weight='length')
                path_nodes = nx.shortest_path(self.graph, origin_node, destination_node, weight='length')
                self.route_cache[cache_key] = (distance, path_nodes)
            except nx.NetworkXNoPath:
                return None, None
                
        return self.route_cache[cache_key]

    
    def process_user_request(self, user_id: int, restaurant_id: int) -> bool:
        """
        Finalizes an order. Assumes visibility/distance has already 
        been verified by the 'Discovery' layer.
        """
        user = self.users.get(user_id)
        res = self.restaurants.get(restaurant_id)
        
        if not user or not res:
            return False

        # If the restaurant has kitchen capacity, we fulfill the order
        if res.can_accept_order():
            # Get path data (from cache if possible) to attach to the order
            distance, path = self.get_route_data(user.location, res.location)
            
            if distance is None:
                return False # Safety check if nodes aren't connected

            p_time = res.generate_prep_time()
            new_order = Order(
                order_id=self.order_id_counter,
                user_id=user_id,
                restaurant_id=restaurant_id,
                prep_time=p_time,
                start_time=self.current_time,
                route_to_user=path 
            )
            
            self.orders.append(new_order)
            res.accept_order(new_order)
            self.order_id_counter += 1
            return True
            
        return False

    def dispatch_logic(self):
        """Placeholder for future matching and routing algorithms"""
        pass

    def run_tick(self):
        """Advances the simulation by one time step"""
        self.current_time += self.step_size
        
        # Update Restaurant states (cooking logic)
        for res in self.restaurants.values():
            res.update_preparing_orders_to_ready(self.current_time)
        
        # Update Driver states (movement and delivery logic)
        for driver in self.drivers:
            # driver.update_position(self.current_time, self.step_size)
            pass
            
        self.dispatch_logic()

    def run_until(self, end_time):
        """Executes simulation ticks until reaching the specified end time"""
        while self.current_time < end_time:
            self.run_tick()

    def get_preparing_orders(self):
        """Returns IDs of orders currently in the PREPARING state"""
        return [o.id for o in self.orders if o.status == "PREPARING"]

