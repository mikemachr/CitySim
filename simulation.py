from __future__ import annotations
import random
import numpy as np
from collections import deque

from environment import Environment
from agents import Driver, DriverEvent, Order, Restaurant, User
from policies.base import DispatchPolicy, RepositioningPolicy
from policies.dispatch import HungarianPolicy
from policies.repositioning import StaticPolicy


class Simulation:
    """
    Thin orchestrator. Owns the clock and agent registries.
    All routing delegated to Environment.
    All dispatch/repositioning logic delegated to policies.
    Reacts to DriverEvents — applies timestamps and restaurant side-effects.
    """

    def __init__(
        self,
        env: Environment,
        dispatch_policy: DispatchPolicy       = None,
        repositioning_policy: RepositioningPolicy = None,
        step_size: float = 10,
        dispatch_interval: float = 15,       # seconds, for batch policies
        start_hour: float = 0.0 
    ):
        self.env                  = env
        self.dispatch_policy      = dispatch_policy or HungarianPolicy()
        self.repositioning_policy = repositioning_policy or StaticPolicy()
        self.step_size            = step_size
        self.dispatch_interval    = dispatch_interval
        self.start_hour = start_hour  # hour of day the simulation begins (0–23)
        self.current_time: float  = 0.0
        self.last_dispatch_time: float = 0.0

        self.restaurants: dict[int, Restaurant] = {}
        self.users:       dict[int, User]       = {}
        self.drivers:     dict[int, Driver]     = {}
        self.orders:      dict[int, Order]      = {}
        self.order_id_counter: int = 1

        self.pending_orders: deque[int] = deque()
        self._pending_set:   set[int]   = set()
        self.idle_drivers:   set[int]   = set()

        self._active_user_ids: set[int] = set() # For guarding against same user placing more than one order at the time

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
    # Order creation
    # ------------------------------------------------------------------

    def process_user_request(self, user_id: int, restaurant_id: int) -> bool:
        user = self.users.get(user_id)
        res  = self.restaurants.get(restaurant_id)
        if not user or not res or not res.can_accept_order():
            return False

        distance, path = self.env.get_route(res.location, user.location)
        if distance is None:
            return False

        order = Order(
            order_id=self.order_id_counter,
            user_id=user_id,
            restaurant_id=restaurant_id,
            prep_time=res.generate_prep_time(),
            start_time=self.current_time,
            route_to_user=path,
        )
        self.orders[self.order_id_counter] = order
        res.accept_order(order)
        self.pending_orders.append(order.id)
        self._pending_set.add(order.id)
        self.order_id_counter += 1
        self._active_user_ids.add(user_id)
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_tick(self):
        self.current_time += self.step_size

        # 1. Restaurants: advance cooking
        for res in self.restaurants.values():
            res.update_preparing_orders_to_ready(self.current_time)

        # 2. Drivers: advance movement, collect events
        for driver in self.drivers.values():
            events = driver.update_position(self.step_size, self.env)
            self._handle_driver_events(driver, events)

        # 3. Dispatch on interval
        if self.current_time - self.last_dispatch_time >= self.dispatch_interval:
            self._run_dispatch()
            self.last_dispatch_time = self.current_time

        # 4. Reposition newly idle drivers
        self._run_repositioning()

    def run_until(self, end_time: float):
        while self.current_time < end_time:
            self.run_tick()

    # ------------------------------------------------------------------
    # Event handler — only place timestamps and side-effects are applied
    # ------------------------------------------------------------------

    def _handle_driver_events(self, driver: Driver, events: list[tuple]):
        for event, order in events:
            if event == DriverEvent.PICKUP_COMPLETE:
                order.pickup_time = self.current_time
                self.restaurants[order.restaurant_id].remove_order(order)

            elif event == DriverEvent.DROPOFF_COMPLETE:
                order.delivered_time = self.current_time
                self._active_user_ids.discard(order.user_id)

            elif event == DriverEvent.WENT_IDLE:
                self.idle_drivers.add(driver.id)

    # ------------------------------------------------------------------
    # Dispatch — fully delegated to policy
    # ------------------------------------------------------------------

    def _run_dispatch(self):
        if not self._pending_set or not self.idle_drivers:
            return

        idle_locations    = {did: self.drivers[did].location for did in self.idle_drivers}
        pending_locations = {
            oid: self.restaurants[self.orders[oid].restaurant_id].location
            for oid in self._pending_set
        }

        assignments = self.dispatch_policy.assign(idle_locations, pending_locations, self.env)

        for driver_id, order_id in assignments:
            order  = self.orders[order_id]
            driver = self.drivers[driver_id]
            driver.speed = get_courier_speed_ms(self.wall_clock_hour)  # update speed dynamically 
            order.driver_id    = driver_id
            order.assigned_time = self.current_time

            self.idle_drivers.discard(driver_id)
            self._pending_set.discard(order_id)
            try:
                self.pending_orders.remove(order_id)
            except ValueError:
                pass

            _, pickup_route = self.env.get_route(driver.location,
                                                   self.restaurants[order.restaurant_id].location)
            driver.assign_order(order, pickup_route or [driver.location])

    # ------------------------------------------------------------------
    # Repositioning — fully delegated to policy
    # ------------------------------------------------------------------

    def _run_repositioning(self):
        if not self.idle_drivers:
            return

        idle_locations = {did: self.drivers[did].location for did in self.idle_drivers}
        sim_state = {
            'current_time':    self.current_time,
            'pending_count':   len(self._pending_set),
            'restaurant_nodes': [r.location for r in self.restaurants.values()],
        }

        targets = self.repositioning_policy.reposition(idle_locations, self.env, sim_state)

        for driver_id, target_node in targets.items():
            driver = self.drivers[driver_id]
            if target_node != driver.location:
                _, route = self.env.get_route(driver.location, target_node)
                if route:
                    driver.assign_route(route)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_nearby_restaurants(self, user_id: int, max_dist: float = 2500) -> list[int]:
        user = self.users.get(user_id)
        if not user:
            return []
        reachable = self.env.get_reachable(user.location, max_dist)
        return [rid for rid, res in self.restaurants.items() if res.location in reachable]

    def get_orders_by_status(self, status: str | list) -> list[int]:
        if isinstance(status, str):
            status = [status]
        return [o.id for o in self.orders.values() if o.status in status]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics_snapshot(self) -> dict:
        delivered = [o for o in self.orders.values() if o.status == 'DELIVERED']

        def safe_mean(vals):
            v = [x for x in vals if x is not None]
            return float(np.mean(v)) if v else None

        n_by_status = {s: 0 for s in ('PREPARING', 'READY', 'PICKED_UP', 'DELIVERED')}
        for o in self.orders.values():
            n_by_status[o.status] = n_by_status.get(o.status, 0) + 1

        return {
            'time':                self.current_time,
            'dispatch_policy':     self.dispatch_policy.__class__.__name__,
            'repositioning_policy': self.repositioning_policy.__class__.__name__,
            'total_orders':        len(self.orders),
            'orders_by_status':    n_by_status,
            'idle_drivers':        len(self.idle_drivers),
            'pending_unassigned':  len(self._pending_set),
            'avg_end_to_end_s':    safe_mean(o.end_to_end_time for o in delivered),
            'avg_food_wait_s':     safe_mean(o.food_wait_time  for o in delivered),
            'avg_dispatch_delay_s': safe_mean(o.dispatch_delay  for o in delivered),
            'n_delivered':         len(delivered),
        }

    @property
    def wall_clock_hour(self) -> float:
        """Current simulated hour of day (0–24)."""
        return (self.start_hour + self.current_time / 3600) % 24
    
    @property
    def wall_clock_display(self) -> str:
        """Returns simulated wall clock as a human-readable string.
        
        Returns:
            str: Formatted time string e.g. 'Day 2  14:35:10'.
        """
        total_seconds = int(self.start_hour * 3600 + self.current_time)
        day     = total_seconds // 86400 + 1
        hour    = (total_seconds % 86400) // 3600
        minute  = (total_seconds % 3600) // 60
        second  = total_seconds % 60
        return f"Day {day}  {hour:02d}:{minute:02d}:{second:02d}"

# ---------------------------------------------------------------------------
# Order generation (Poisson arrivals + MNL restaurant choice)
# ---------------------------------------------------------------------------

def generate_orders(sim: Simulation, rate_per_minute: float):
    """
    Poisson arrivals. Restaurant choice via multinomial logit:
        utility = 1.0 * rating - 0.8 * (distance_km)
    """
    # Source: synthesized from Brazil delivery fee model (Frontiers 2022)
    # and Ma et al. 2024 (Singapore mixed logit).
    # β_rating range: +0.30 to +1.00 per star → use 0.6 midpoint
    # β_distance range: −0.15 to −0.50 per km → use 0.3
    # Ratio β_rating/β_distance implies indifference between
    # +1 star and +2 km, which is reasonable for urban Monterrey.
    ALPHA = 0.6   # per rating point (1–5 scale)
    BETA  = 0.3   # per km of distance

    arrivals = np.random.poisson(rate_per_minute / 60 * sim.step_size)
    if arrivals == 0:
        return

    user_ids = list(sim.users.keys())

    for _ in range(arrivals):
        user_id = random.choice(user_ids)
        user    = sim.users[user_id]
        # Guard: skip if user already has an active order
        if user_id in sim._active_user_ids:
            continue
        #reachable  = sim.env.get_reachable(user.location, cutoff_m=2500)
        # Use cached version instead
        reachable = sim.env.get_reachable_cached(user.location, cutoff_m=2500)
        candidates = [
            (rid, reachable[res.location])
            for rid, res in sim.restaurants.items()
            if res.location in reachable
        ]
        if not candidates:
            continue

        utils = np.array([
            ALPHA * sim.restaurants[rid].rating - BETA * (dist / 1000)
            for rid, dist in candidates
        ])
        utils -= utils.max()
        probs  = np.exp(utils)
        probs /= probs.sum()

        restaurant_id = candidates[np.random.choice(len(candidates), p=probs)][0]
        sim.process_user_request(user_id, restaurant_id)

# ---------------------------------------------------------------------------
# Dynamic order rate
# ---------------------------------------------------------------------------


def get_order_rate(sim: Simulation) -> float:
    """Returns orders/min for Zona Tec based on Monterrey peak patterns.
    
    Calibrated from TomTom + DiDi Food/CANIRAC triangulation.
    Zona Tec assumed ~15-25% of metro order volume given restaurant density.

    Args:
        sim: Running Simulation instance.

    Returns:
        float: Orders per minute for the current simulated hour.
    """
    hour = sim.wall_clock_hour
    if 12.0 <= hour < 14.0:
        return 25.0   # lunch peak
    elif 19.0 <= hour < 22.0:
        return 20.0   # dinner peak
    else:
        return 8.0    # off-peak
# ---------------------------------------------------------------------------
# Dynamic courier speed
# ---------------------------------------------------------------------------
    

def get_courier_speed_ms(hour: float) -> float:
    """Returns mean courier speed in m/s based on Monterrey TomTom data.
    
    Args:
        hour: Current hour of day (0–24 float).
    
    Returns:
        Mean speed in m/s, sampled from a normal with ~10% std.
        Source: TomTom Traffic Index 2025, Monterrey.
    """
    if 12.0 <= hour < 14.0:       # lunch peak
        mean_kmh = 31.0
    elif 18.0 <= hour < 19.0:     # worst congestion
        mean_kmh = 22.0
    elif 19.0 <= hour < 21.0:     # dinner, easing
        mean_kmh = 30.0
    elif 22.0 <= hour or hour < 6.0:  # overnight
        mean_kmh = 50.0
    else:                          # general daytime
        mean_kmh = 32.0
    
    mean_ms = mean_kmh / 3.6
    return max(4.5, random.gauss(mean_ms, mean_ms * 0.10))