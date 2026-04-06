import random
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString
from collections import deque


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class Order:
    """Links restaurants, users, and drivers. Tracks full timing for metrics."""

    def __init__(self, order_id, user_id, restaurant_id, prep_time, start_time, route_to_user):
        self.id = order_id
        self.user_id = user_id
        self.restaurant_id = restaurant_id
        self.driver_id: int | None = None

        # Timing
        self.prep_time = prep_time
        self.start_time = start_time          # tick order was placed
        self.ready_time = start_time + prep_time

        # Filled in as events happen — used for metrics
        self.assigned_time: float | None = None   # when a driver was assigned
        self.pickup_time: float | None = None     # when driver finished pickup service
        self.delivered_time: float | None = None  # when dropoff service ended

        # Statuses: 'PREPARING' -> 'READY' -> 'PICKED_UP' -> 'DELIVERED'
        self.status = 'PREPARING'
        self.route_to_user = route_to_user

    # ------------------------------------------------------------------
    # Derived metrics (only valid after delivery)
    # ------------------------------------------------------------------

    @property
    def end_to_end_time(self) -> float | None:
        """Total time from order placed to delivered (seconds)."""
        if self.delivered_time is None:
            return None
        return self.delivered_time - self.start_time

    @property
    def food_wait_time(self) -> float | None:
        """Time food sat ready before a driver picked it up (seconds)."""
        if self.pickup_time is None:
            return None
        ready = max(self.ready_time, self.assigned_time or self.ready_time)
        return max(0.0, self.pickup_time - ready)

    @property
    def dispatch_delay(self) -> float | None:
        """Time between order placed and driver assigned (seconds)."""
        if self.assigned_time is None:
            return None
        return self.assigned_time - self.start_time


# ---------------------------------------------------------------------------
# Restaurant
# ---------------------------------------------------------------------------

class Restaurant:
    """Fulfils orders. Has rating and capacity; generates lognormal prep times."""

    def __init__(self, restaurant_id: int, location: int, rating: float,
                 capacity: int, avg_prep_time: float, service_radius: float,
                 enabled=True):
        self.id = restaurant_id
        self.location = location       # node id in graph
        self.rating = rating           # 1.0 - 5.0, used in restaurant choice model
        self.capacity = capacity
        self.avg_prep_time = avg_prep_time   # seconds
        self.service_radius = service_radius  # max delivery distance (meters)
        self.active_orders: list[Order] = []
        self.enabled = enabled

    def can_accept_order(self) -> bool:
        if not self.enabled:
            return False
        if len(self.active_orders) >= self.capacity:
            return False
        return random.random() > 0.01   # 1% random rejection

    def generate_prep_time(self) -> float:
        sigma = 0.5
        mu = np.log(self.avg_prep_time) - (sigma ** 2) / 2
        return random.lognormvariate(mu, sigma)

    def update_preparing_orders_to_ready(self, current_time: float):
        for order in self.active_orders:
            if order.status == 'PREPARING' and current_time >= order.ready_time:
                order.status = 'READY'

    def accept_order(self, order: Order):
        self.active_orders.append(order)
        self._sync_enabled_status()

    def remove_order(self, order: Order):
        if order in self.active_orders:
            self.active_orders.remove(order)
            self._sync_enabled_status()

    def _sync_enabled_status(self):
        self.enabled = len(self.active_orders) < self.capacity


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class User:
    def __init__(self, user_id: int, location: int):
        self.user_id = user_id
        self.location = location  # node id in graph


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class Driver:
    """
    Mobile agent. Moves along graph edges using distance-based interpolation.
    Maintains a FIFO queue of orders.
    """

    def __init__(self, driver_id: int, location_node: int, speed: float | None = None):
        self.id = driver_id
        self.location = location_node
        self.coords = (0.0, 0.0)

        # Movement state
        self.current_route: list[int] = []
        self.distance_on_edge: float = 0.0
        self.current_edge: tuple | None = None

        # Logistics state
        self.order_queue: deque[Order] = deque()
        self.active_order: Order | None = None
        self.status = 'IDLE'
        self.service_remaining: float = 0.0
        self.service_type: str | None = None

        # Speed (m/s)
        if speed is None:
            sampled = random.gauss(6.9, 1.0)
            self.speed = max(4.5, sampled)
        else:
            self.speed = speed

    # ------------------------------------------------------------------
    # Service time generators
    # ------------------------------------------------------------------

    def generate_pickup_service_time(self) -> float:
        mu = np.log(75) - (0.45 ** 2) / 2
        return random.lognormvariate(mu, 0.45)

    def generate_dropoff_service_time(self) -> float:
        mu = np.log(130) - (0.55 ** 2) / 2
        return random.lognormvariate(mu, 0.55)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def add_order(self, order: Order, simulation: 'Simulation'):
        self.order_queue.append(order)
        if self.status == 'IDLE':
            self._start_next_task(simulation)

    def _start_next_task(self, simulation: 'Simulation'):
        if not self.order_queue:
            self.status = 'IDLE'
            self.active_order = None
            simulation.idle_drivers.add(self.id)
            return

        self.active_order = self.order_queue[0]

        if self.active_order.status in ('PREPARING', 'READY'):
            self.status = 'PICKING_UP'
            res = simulation.restaurants[self.active_order.restaurant_id]
            _, pickup_path = simulation.get_route_data(self.location, res.location)
            self.set_route(pickup_path)

        elif self.active_order.status == 'PICKED_UP':
            self.status = 'DELIVERING'
            self.set_route(self.active_order.route_to_user)

    def set_route(self, route_nodes):
        if not route_nodes or len(route_nodes) < 2:
            self.current_route = []
            self.current_edge = None
            return
        self.current_route = list(route_nodes)
        self.current_edge = (self.current_route[0], self.current_route[1])
        self.distance_on_edge = 0.0
        self.location = route_nodes[0]

    # ------------------------------------------------------------------
    # Movement engine
    # ------------------------------------------------------------------

    def update_position(self, step_size: float, simulation: 'Simulation'):
        # Handle service delays
        if self.status in ('PICKUP_SERVICE', 'DROPOFF_SERVICE'):
            self.service_remaining -= step_size
            if self.service_remaining <= 0:
                if self.service_type == 'PICKUP':
                    res = simulation.restaurants[self.active_order.restaurant_id]
                    res.remove_order(self.active_order)
                    self.active_order.status = 'PICKED_UP'
                    self.active_order.pickup_time = simulation.current_time
                    self._start_next_task(simulation)
                elif self.service_type == 'DROPOFF':
                    self.active_order.status = 'DELIVERED'
                    self.active_order.delivered_time = simulation.current_time
                    self.order_queue.popleft()
                    self.status = 'IDLE'
                    self._start_next_task(simulation)
            return

        if not self.current_route or len(self.current_route) < 2:
            if self.status == 'PICKING_UP' and self.active_order and self.active_order.status == 'READY':
                self._handle_arrival(simulation)
            return

        self.distance_on_edge += self.speed * step_size

        while len(self.current_route) >= 2:
            u, v = self.current_edge
            edge_data = simulation.graph.get_edge_data(u, v)[0]
            edge_len = edge_data['length']

            if self.distance_on_edge < edge_len:
                break

            self.distance_on_edge -= edge_len
            self.current_route.pop(0)

            if len(self.current_route) < 2:
                self.location = v
                self._handle_arrival(simulation)
                return

            self.current_edge = (self.current_route[0], self.current_route[1])

        self._update_coords(simulation.graph)

    def _handle_arrival(self, simulation: 'Simulation'):
        if self.status == 'PICKING_UP':
            self.status = 'PICKUP_SERVICE'
            self.service_remaining = self.generate_pickup_service_time()
            self.service_type = 'PICKUP'
        elif self.status == 'DELIVERING':
            self.status = 'DROPOFF_SERVICE'
            self.service_remaining = self.generate_dropoff_service_time()
            self.service_type = 'DROPOFF'

    def _update_coords(self, graph):
        if not self.current_edge:
            return
        u, v = self.current_edge
        edge_data = graph.get_edge_data(u, v)[0]
        edge_len = edge_data['length']
        line = edge_data.get('geometry', LineString([
            (graph.nodes[u]['x'], graph.nodes[u]['y']),
            (graph.nodes[v]['x'], graph.nodes[v]['y']),
        ]))
        if edge_len > 0:
            fraction = self.distance_on_edge / edge_len
            point = line.interpolate(min(fraction, 1.0), normalized=True)
            self.coords = (point.y, point.x)   # (lat, lon)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class Simulation:
    """
    Discrete-time simulation of a food delivery platform.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Road network.
    step_size : float
        Seconds per tick (default 1).
    dispatch_mode : str
        'greedy'    - FIFO, assigns on arrival, fires every tick.
        'hungarian' - batch bipartite matching, fires every dispatch_interval seconds.
    dispatch_interval : float
        Seconds between batch dispatches (hungarian mode only).
    pickup_radius : float
        Max distance (meters) a driver is considered for an order.
    """

    DISPATCH_MODES = ('greedy', 'hungarian')

    def __init__(self, graph, step_size: float = 1,
                 dispatch_mode: str = 'hungarian',
                 dispatch_interval: float = 15,
                 pickup_radius: float = 3000):

        if dispatch_mode not in self.DISPATCH_MODES:
            raise ValueError(f"dispatch_mode must be one of {self.DISPATCH_MODES}")

        self.graph = graph
        self.step_size = step_size
        self.current_time: float = 0.0
        self.dispatch_mode = dispatch_mode
        self.dispatch_interval = dispatch_interval
        self.pickup_radius = pickup_radius
        self.last_dispatch_time: float = 0.0

        self.restaurants: dict[int, Restaurant] = {}
        self.users: dict[int, User] = {}
        self.drivers: dict[int, Driver] = {}
        self.orders: dict[int, Order] = {}
        self.order_id_counter: int = 1

        self.route_cache: dict[tuple, tuple] = {}

        # pending_orders deque preserves arrival order for greedy dispatch.
        # _pending_set gives O(1) membership checks for hungarian cleanup.
        self.pending_orders: deque[int] = deque()
        self._pending_set: set[int] = set()

        self.idle_drivers: set[int] = set()

    # ------------------------------------------------------------------
    # Entity registration
    # ------------------------------------------------------------------

    def add_restaurant(self, restaurant: Restaurant):
        self.restaurants[restaurant.id] = restaurant

    def add_user(self, user: User):
        self.users[user.user_id] = user

    def add_driver(self, driver: Driver):
        self.drivers[driver.id] = driver
        self.idle_drivers.add(driver.id)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def get_route_data(self, origin_node: int, destination_node: int):
        """Returns (distance_m, path_nodes). Caches results."""
        if origin_node == destination_node:
            return 0.0, [origin_node]

        cache_key = (origin_node, destination_node)
        if cache_key not in self.route_cache or self.route_cache[cache_key][1] is None:
            try:
                distance = nx.shortest_path_length(
                    self.graph, origin_node, destination_node, weight='length')
                path = nx.shortest_path(
                    self.graph, origin_node, destination_node, weight='length')
                self.route_cache[cache_key] = (distance, path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None, None

        return self.route_cache[cache_key]

    # ------------------------------------------------------------------
    # Order creation
    # ------------------------------------------------------------------

    def process_user_request(self, user_id: int, restaurant_id: int) -> bool:
        """Creates an order if the restaurant can accept it."""
        user = self.users.get(user_id)
        res = self.restaurants.get(restaurant_id)
        if not user or not res:
            return False

        if not res.can_accept_order():
            return False

        distance, path = self.get_route_data(res.location, user.location)
        if distance is None:
            return False

        new_order = Order(
            order_id=self.order_id_counter,
            user_id=user_id,
            restaurant_id=restaurant_id,
            prep_time=res.generate_prep_time(),
            start_time=self.current_time,
            route_to_user=path,
        )
        self.orders[self.order_id_counter] = new_order
        res.accept_order(new_order)

        self.pending_orders.append(new_order.id)
        self._pending_set.add(new_order.id)

        self.order_id_counter += 1
        return True

    # ------------------------------------------------------------------
    # Dispatch strategies (private)
    # ------------------------------------------------------------------

    def _dispatch_greedy(self):
        """
        FIFO greedy: pairs the next pending order with any idle driver.
        No optimisation — first come, first served on both sides.
        """
        while self.pending_orders and self.idle_drivers:
            order_id = self.pending_orders.popleft()
            self._pending_set.discard(order_id)
            order = self.orders[order_id]

            if order.driver_id is not None:
                continue

            driver_id = self.idle_drivers.pop()
            order.driver_id = driver_id
            order.assigned_time = self.current_time
            self.drivers[driver_id].add_order(order, self)

    def _dispatch_hungarian(self):
        """
        Batch bipartite matching via the Hungarian algorithm.
        Minimises total driver-to-restaurant travel distance across all
        idle-driver / unassigned-order pairs reachable within pickup_radius.
        """
        driver_ids = [d.id for d in self.drivers.values() if d.status == 'IDLE']
        order_ids = [
            o.id for o in self.orders.values()
            if o.status in ('PREPARING', 'READY') and o.driver_id is None
        ]

        if not driver_ids or not order_ids:
            return

        cost_matrix = np.full((len(driver_ids), len(order_ids)), np.inf)

        for i, driver_id in enumerate(driver_ids):
            driver = self.drivers[driver_id]
            reachable = nx.single_source_dijkstra_path_length(
                self.graph, driver.location,
                cutoff=self.pickup_radius, weight='length'
            )
            for j, order_id in enumerate(order_ids):
                res_loc = self.restaurants[self.orders[order_id].restaurant_id].location
                if res_loc in reachable:
                    cost_matrix[i, j] = reachable[res_loc]

        valid_rows = ~np.all(np.isinf(cost_matrix), axis=1)
        valid_cols = ~np.all(np.isinf(cost_matrix), axis=0)
        filtered = cost_matrix[np.ix_(valid_rows, valid_cols)]

        if filtered.size == 0:
            return

        row_ind, col_ind = linear_sum_assignment(filtered)
        row_ind = np.where(valid_rows)[0][row_ind]
        col_ind = np.where(valid_cols)[0][col_ind]

        for r, c in zip(row_ind, col_ind):
            driver_id = driver_ids[r]
            order_id = order_ids[c]
            order = self.orders[order_id]
            order.driver_id = driver_id
            order.assigned_time = self.current_time

            if order_id in self._pending_set:
                self._pending_set.discard(order_id)
                try:
                    self.pending_orders.remove(order_id)
                except ValueError:
                    pass

            self.drivers[driver_id].add_order(order, self)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_tick(self):
        """Advances simulation by one step_size."""
        self.current_time += self.step_size

        for res in self.restaurants.values():
            res.update_preparing_orders_to_ready(self.current_time)

        for driver in self.drivers.values():
            driver.update_position(self.step_size, self)

        if self.dispatch_mode == 'greedy':
            self._dispatch_greedy()
        else:
            if self.current_time - self.last_dispatch_time >= self.dispatch_interval:
                self._dispatch_hungarian()
                self.last_dispatch_time = self.current_time

    def run_until(self, end_time: float):
        while self.current_time < end_time:
            self.run_tick()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_nearby_restaurants(self, user_id: int, max_dist: float = 2500) -> list[int]:
        user = self.users.get(user_id)
        if not user or not self.graph:
            return []
        reachable = nx.single_source_dijkstra_path_length(
            self.graph, user.location, cutoff=max_dist, weight='length'
        )
        return [res_id for res_id, res in self.restaurants.items()
                if res.location in reachable]

    def get_orders_by_status(self, status: str | list) -> list[int]:
        if isinstance(status, str):
            status = [status]
        return [o.id for o in self.orders.values() if o.status in status]

    # ------------------------------------------------------------------
    # Metrics snapshot
    # ------------------------------------------------------------------

    def metrics_snapshot(self) -> dict:
        """
        Returns a dict of current KPIs.
        Time-based averages are computed over all delivered orders.
        """
        delivered = [o for o in self.orders.values() if o.status == 'DELIVERED']

        def safe_mean(vals):
            v = [x for x in vals if x is not None]
            return float(np.mean(v)) if v else None

        n_by_status = {s: 0 for s in ('PREPARING', 'READY', 'PICKED_UP', 'DELIVERED')}
        for o in self.orders.values():
            n_by_status[o.status] = n_by_status.get(o.status, 0) + 1

        return {
            'time': self.current_time,
            'dispatch_mode': self.dispatch_mode,
            'total_orders': len(self.orders),
            'orders_by_status': n_by_status,
            'idle_drivers': len(self.idle_drivers),
            'pending_unassigned': len(self._pending_set),
            'avg_end_to_end_s': safe_mean(o.end_to_end_time for o in delivered),
            'avg_food_wait_s': safe_mean(o.food_wait_time for o in delivered),
            'avg_dispatch_delay_s': safe_mean(o.dispatch_delay for o in delivered),
            'n_delivered': len(delivered),
        }


# ---------------------------------------------------------------------------
# Order generation (Poisson arrivals + softmax restaurant choice)
# ---------------------------------------------------------------------------

def generate_orders(sim: Simulation, rate_per_minute: float):
    """
    Generates new orders using a Poisson arrival process.

    Restaurant choice uses a softmax over a utility score:
        utility = alpha * rating - beta * (distance_m / 1000)

    This is a multinomial logit (MNL) model — standard in discrete
    choice / transportation OR literature.

    Parameters
    ----------
    sim : Simulation
    rate_per_minute : float
        Expected orders per minute across all users (lambda for Poisson).
    """
    ALPHA = 1.0   # weight on rating (scale 1-5)
    BETA = 0.8    # penalty per km of distance

    arrivals = np.random.poisson(rate_per_minute / 60 * sim.step_size)
    if arrivals == 0:
        return

    user_ids = list(sim.users.keys())

    for _ in range(arrivals):
        user_id = random.choice(user_ids)
        user = sim.users[user_id]

        reachable = nx.single_source_dijkstra_path_length(
            sim.graph, user.location, cutoff=2500, weight='length'
        )
        candidates = [
            (res_id, reachable[res.location])
            for res_id, res in sim.restaurants.items()
            if res.location in reachable
        ]
        if not candidates:
            continue

        utils = np.array([
            ALPHA * sim.restaurants[rid].rating - BETA * (dist / 1000)
            for rid, dist in candidates
        ])
        utils -= utils.max()   # numerical stability before exp
        probs = np.exp(utils)
        probs /= probs.sum()

        restaurant_id = candidates[np.random.choice(len(candidates), p=probs)][0]
        sim.process_user_request(user_id, restaurant_id)
