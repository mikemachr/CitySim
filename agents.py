import random
import networkx as nx
from shapely.geometry import LineString
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
class Order:
    """Links restaurants, users and drivers"""
    def __init__(self, order_id, user_id, restaurant_id, prep_time, start_time,route_to_user):
        self.id = order_id
        self.user_id = user_id
        self.restaurant_id = restaurant_id
        self.driver_id: int|None = None
        
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
        """Prep time sampled from lognormal distribution"""
        mean = self.avg_prep_time

        # Convert mean to lognormal parameters
        sigma = 0.5
        mu = np.log(mean) - (sigma**2)/2

        return random.lognormvariate(mu, sigma)
    
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

class Driver:
    """
    Mobile agent that manages a queue of orders and moves continuously 
    along the graph edges using distance-based interpolation.
    """
    def __init__(self, driver_id: int, location_node: int, speed: float|None = None):
        self.id = driver_id
        self.location = location_node
        self.coords = (0.0, 0.0)
        
        # Movement state
        self.current_route = []
        self.distance_on_edge = 0.0
        self.current_edge = None
        
        # Logistics state
        self.order_queue = deque() 
        self.active_order:Order|None = None
        self.status = 'IDLE' # 'IDLE', 'PICKING_UP', 'DELIVERING'
        self.service_remaining = 0
        self.service_type = None
        # Effective driving speed (m/s)
        # If not provided, sample from normal distribution
        if speed is None:
            sampled = random.gauss(6.9, 1.0)
            self.speed = max(4.5, sampled)
        else:
            self.speed = speed

    def generate_pickup_service_time(self):
        """Pickup delay (seconds)"""
        mean = 90
        return random.expovariate(1.0 / mean)

    def generate_dropoff_service_time(self):
        """Dropoff delay (seconds)"""
        mean = 120
        return random.expovariate(1.0 / mean)
    def add_order(self, order, simulation):
        """Adds an order to the FIFO queue and triggers movement if IDLE."""
        self.order_queue.append(order)
        if self.status == 'IDLE':
            self._start_next_task(simulation)

    def _start_next_task(self, simulation):

        if not self.order_queue:
            self.status = 'IDLE'
            self.active_order = None
            simulation.idle_drivers.add(self.id)
            return

        self.active_order = self.order_queue[0]

        # If order not picked up → go to restaurant
        if self.active_order.status in ['PREPARING', 'READY']:

            self.status = 'PICKING_UP'

            res = simulation.restaurants[self.active_order.restaurant_id]

            _, pickup_path = simulation.get_route_data(
                self.location,
                res.location
            )

            self.set_route(pickup_path)

        # If order already picked up → deliver
        elif self.active_order.status == 'PICKED_UP':

            self.status = 'DELIVERING'

            self.set_route(self.active_order.route_to_user)

    def set_route(self, route_nodes):
        """Initializes a route and prepares edge variables."""
        if not route_nodes or len(route_nodes) < 2:
            self.current_route = []
            self.current_edge = None
            return
        self.current_route = list(route_nodes)
        self.current_edge = (self.current_route[0], self.current_route[1])
        self.distance_on_edge = 0.0
        self.location = route_nodes[0]

    def update_position(self, step_size, simulation):
        """
        Movement engine with optimized route transition and safety checks.
        """
        # Handle service delays
        if self.status in ['PICKUP_SERVICE', 'DROPOFF_SERVICE']:
            self.service_remaining -= step_size

            if self.service_remaining <= 0:

                if self.service_type == "PICKUP":

                    res = simulation.restaurants[self.active_order.restaurant_id]
                    res.remove_order(self.active_order)

                    self.active_order.status = 'PICKED_UP'

                    # start delivery route

                    self._start_next_task(simulation)

                elif self.service_type == "DROPOFF":

                    self.active_order.status = 'DELIVERED'
                    self.order_queue.popleft()
                    self.status = 'IDLE'
                    self._start_next_task(simulation)

            return
        # 1. Early exit if no route exists
        if not self.current_route or len(self.current_route) < 2:
            if self.status == 'PICKING_UP' and self.active_order and self.active_order.status == 'READY':
                self._handle_arrival(simulation)
            return

        # 2. Advance total distance for this tick
        self.distance_on_edge += self.speed * step_size
        
        # 3. Process node transitions
        # We use a while loop to handle cases where step_size covers multiple edges

        while len(self.current_route) >= 2:
            u, v = self.current_edge
            edge_data = simulation.graph.get_edge_data(u, v)[0]
            edge_len = edge_data['length']

            # If we haven't finished the current edge, stop transitioning
            if self.distance_on_edge < edge_len:
                break
            
            # If we finished the edge, subtract its length and move to the next node
            self.distance_on_edge -= edge_len
            self.current_route.pop(0)
            
            # Check if we just reached the end of the entire route
            if len(self.current_route) < 2:
                self.location = v
                self._handle_arrival(simulation)
                return

            # Set up the next edge for the next iteration of the while loop
            self.current_edge = (self.current_route[0], self.current_route[1])

        # 4. Update the visual coordinates based on final position in this tick
        self._update_coords(simulation.graph)

    def _handle_arrival(self, simulation):
        """State transition logic upon reaching a destination."""
        if self.status == 'PICKING_UP':

            # Always start pickup service when arriving
            self.status = 'PICKUP_SERVICE'
            self.service_remaining = self.generate_pickup_service_time()
            self.service_type = "PICKUP"

        elif self.status == 'DELIVERING':

            self.status = 'DROPOFF_SERVICE'
            self.service_remaining = self.generate_dropoff_service_time()
            self.service_type = "DROPOFF"

    def _update_coords(self, graph):
            """Updates coordinates using normalized interpolation to bridge meters and degrees."""
            if not self.current_edge:
                return
                
            u, v = self.current_edge
            edge_data = graph.get_edge_data(u, v)[0]
            edge_len = edge_data['length'] # Meters
            
            line = edge_data.get('geometry', LineString([
                (graph.nodes[u]['x'], graph.nodes[u]['y']), 
                (graph.nodes[v]['x'], graph.nodes[v]['y'])
            ]))
            
            # Calculate progress as a fraction (0.0 to 1.0)
            # This prevents 'teleportation' by ignoring raw coordinate units
            if edge_len > 0:
                fraction = self.distance_on_edge / edge_len
                # Use normalized=True to treat the line length as 1.0
                point = line.interpolate(min(fraction, 1.0), normalized=True)
                self.coords = (point.y, point.x) # (lat, lon)

class Simulation:
    def __init__(self,graph, step_size=1):
        self.current_time = 0       # Total seconds elapsed
        self.step_size = step_size  # Duration of each tick
        self.restaurants: dict[int,Restaurant] = {} # Dictionary for O(1) restaurant lookup
        self.users : dict[int,User] = {} # Dictionary for O(1) restaurant lookup
        self.drivers: dict[int, Driver] = {} # Dict for O(1) lookup
        self.orders: dict[int,Order] = {}            # Global order history for analytics
        self.order_id_counter = 1
        self.route_cache = {}
        self.graph = graph
        self.pending_orders: deque[int] = deque()
        self.idle_drivers: set[int] = set()
        self.dispatch_interval = 15   # seconds between batch dispatches
        self.last_dispatch_time = 0
        self.pickup_radius = 3000  # meters

    def add_restaurant(self, restaurant: Restaurant):
        self.restaurants[restaurant.id] = restaurant

    def add_user(self,user: User):
        self.users[user.user_id] = user

    def add_driver(self, driver: Driver):
            self.drivers[driver.id] = driver
            self.idle_drivers.add(driver.id)

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
        
        # Check if key is missing OR if the path (index 1) is missing
        if cache_key not in self.route_cache or self.route_cache[cache_key][1] is None:
            try:
                distance = nx.shortest_path_length(self.graph, origin_node, destination_node, weight='length')
                path_nodes = nx.shortest_path(self.graph, origin_node, destination_node, weight='length')
                self.route_cache[cache_key] = (distance, path_nodes)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
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
            distance, path = self.get_route_data(res.location,user.location)
            
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
            # Add order to simulation object 
            self.orders[self.order_id_counter] =new_order
            # Add order to restaurant 
            res.accept_order(new_order)
            self.pending_orders.append(new_order.id)
            self.order_id_counter += 1

            return True
            
        return False

    def dispatch_logic(self):
        """
        Event-driven dispatcher.

        Assigns orders to idle drivers without scanning the whole
        system every tick. Dispatch happens only when either
        a new order appears or a driver becomes available.
        """

        while self.pending_orders and self.idle_drivers:

            order_id = self.pending_orders.popleft()
            order = self.orders[order_id]

            # Safety check (order might have been cancelled in future versions)
            if order.driver_id is not None:
                continue

            driver_id = self.idle_drivers.pop()
            driver = self.drivers[driver_id]

            order.driver_id = driver_id
            driver.add_order(order, self)

    def batch_dispatch(self):
        """
        Assign drivers to orders using batch bipartite matching (Hungarian algorithm).
        Drivers are matched to restaurant pickup locations minimizing travel distance.
        """

        import numpy as np
        import networkx as nx
        from scipy.optimize import linear_sum_assignment

        # Collect idle drivers
        driver_ids = [
            d.id for d in self.drivers.values()
            if d.status == "IDLE"
        ]

        # Orders eligible for pickup
        order_ids = [
        o.id for o in self.orders.values()
        if o.status in ["PREPARING", "READY"] and o.driver_id is None
        ]


        if not driver_ids or not order_ids:
            return

        n_drivers = len(driver_ids)
        n_orders = len(order_ids)

        # Initialize cost matrix
        cost_matrix = np.full((n_drivers, n_orders), np.inf)

        # Build cost matrix using spatial filtering
        for i, driver_id in enumerate(driver_ids):

            driver = self.drivers[driver_id]

            # Dijkstra search from driver position
            reachable_nodes = nx.single_source_dijkstra_path_length(
                self.graph,
                driver.location,
                cutoff=self.pickup_radius,
                weight="length"
            )

            for j, order_id in enumerate(order_ids):

                order = self.orders[order_id]
                res = self.restaurants[order.restaurant_id]

                if res.location in reachable_nodes:
                    cost_matrix[i, j] = reachable_nodes[res.location]

        # --- Remove infeasible rows / columns ---

        valid_driver_rows = ~np.all(np.isinf(cost_matrix), axis=1)
        valid_order_cols = ~np.all(np.isinf(cost_matrix), axis=0)

        filtered_matrix = cost_matrix[np.ix_(valid_driver_rows, valid_order_cols)]

        if filtered_matrix.size == 0:
            return

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(filtered_matrix)

        # Map back to original indices
        driver_indices = np.where(valid_driver_rows)[0]
        order_indices = np.where(valid_order_cols)[0]

        row_ind = driver_indices[row_ind]
        col_ind = order_indices[col_ind]

        # Execute assignments
        for r, c in zip(row_ind, col_ind):

            driver_id = driver_ids[r]
            order_id = order_ids[c]

            driver = self.drivers[driver_id]
            order = self.orders[order_id]

            # ---- critical missing line ----
            order.driver_id = driver_id

            # remove from pending queue if present
            if order_id in self.pending_orders:
                try:
                    self.pending_orders.remove(order_id)
                except ValueError:
                    pass

            driver.add_order(order, self)

    def run_tick(self):
        """Advances the simulation by one time step"""
        self.current_time += self.step_size
        
        # Update Restaurant states (cooking logic)
        for res in self.restaurants.values():
            res.update_preparing_orders_to_ready(self.current_time)
        
        # Update Driver states (movement and delivery logic)
        for driver in self.drivers.values():
            driver.update_position(self.step_size, self)

        # Run batch dispatch periodically
        if self.current_time - self.last_dispatch_time >= self.dispatch_interval:
            self.batch_dispatch()
            self.last_dispatch_time = self.current_time
            

    def get_nearby_restaurants(self, user_id, max_dist=2500):
        user = self.users.get(user_id)
        if not user or not self.graph:
            return []

        # This finds distances from User -> All reachable nodes
        reachable_nodes = nx.single_source_dijkstra_path_length(
            self.graph, 
            user.location, 
            cutoff=max_dist, 
            weight='length'
        )

        available_restaurants = []
        for res_id, res in self.restaurants.items():
            if res.location in reachable_nodes:
                # DO NOT set the path to None in the route_cache here.
                # If you do, the next call will return None instead of calculating the path.
                available_restaurants.append(res_id)
        
        return available_restaurants

    def run_until(self, end_time):
        """Executes simulation ticks until reaching the specified end time"""
        while self.current_time < end_time:
            self.run_tick()

    def get_orders_by_status(self,status: str|list):
        """Returns IDs of orders currently in the requested state"""
        if type(status)==str:
            status = [status]
        return [o.id for o in self.orders.values() if o.status in status]
    