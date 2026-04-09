"""agents.py
Contains the clases used for the agents inside the simulation
  - Drivers
  - DriverEvents
  - Orders
  - Users
  - Restaurants 
"""
from __future__ import annotations
import random
import numpy as np
from shapely.geometry import LineString
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import Environment


# ---------------------------------------------------------------------------
# Driver events — emitted instead of calling back into Simulation
# ---------------------------------------------------------------------------

class DriverEvent(Enum):
    ARRIVED_AT_RESTAURANT = "ARRIVED_AT_RESTAURANT"
    PICKUP_COMPLETE       = "PICKUP_COMPLETE"
    ARRIVED_AT_USER       = "ARRIVED_AT_USER"
    DROPOFF_COMPLETE      = "DROPOFF_COMPLETE"
    WENT_IDLE             = "WENT_IDLE"


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class Order:
    """Links restaurants, users, and drivers. Tracks full timing for metrics."""

    def __init__(self, order_id, user_id, restaurant_id, prep_time, start_time, route_to_user):
        self.id            = order_id
        self.user_id       = user_id
        self.restaurant_id = restaurant_id
        self.driver_id: int | None = None

        self.prep_time  = prep_time
        self.start_time = start_time
        self.ready_time = start_time + prep_time

        self.assigned_time:  float | None = None
        self.pickup_time:    float | None = None
        self.delivered_time: float | None = None

        self.status : str        = 'PREPARING'
        self.route_to_user = route_to_user
        
        self.prior_rating: float | None = None  # restaurant.rating snapshotted at order creation
        self.rating:      int | None   = None  # discrete 1-5, set post-delivery
        self.rated_time:  float | None = None  # sim time when rating was submitted

    @property
    def end_to_end_time(self) -> float | None:
        """Returns the total elapsed time from preparation to delivery."""
        if self.delivered_time is None:
            return None
        return self.delivered_time - self.start_time

    @property
    def food_wait_time(self) -> float | None:
        """
        Returns the amount of time the user spent waiting for pickup.

        Note:
            This assumes that the driver was assigned at or before ready_time.
        """
        if self.pickup_time is None:
            return None
        ready = max(self.ready_time, self.assigned_time or self.ready_time)
        return max(0.0, self.pickup_time - ready)

    @property
    def time_to_assign(self) -> float | None:
        """Seconds from order placement to driver assignment.
        Includes prep time, so this is not a measure of dispatcher speed alone."""
        if self.assigned_time is None:
            return None
        return self.assigned_time - self.start_time

    @property
    def dispatch_delay(self) -> float | None:
        """Seconds from food ready to driver assignment.
        Negative means the driver was assigned before the food finished (good).
        Positive means the food waited unassigned after being ready."""
        if self.assigned_time is None:
            return None
        return self.assigned_time - self.ready_time


# ---------------------------------------------------------------------------
# Restaurant
# ---------------------------------------------------------------------------

class Restaurant:
    """Fulfils orders. Generates lognormal prep times."""

    def __init__(self, restaurant_id: int, location: int, rating: float,
                 capacity: int, avg_prep_time: float, service_radius: float,
                 enabled: bool = True):
        self.id             = restaurant_id
        self.location       = location
        self.rating         = rating
        self.capacity       = capacity
        self.avg_prep_time  = avg_prep_time
        self.service_radius = service_radius
        self.active_orders: list[Order] = []
        self.enabled        = enabled
        # Rating accumulators — O(1) incremental average
        self._rating_sum:   float = 0.0
        self._rating_count: int   = 0

    def can_accept_order(self) -> bool:
        """
        Check whether the restaurant can accept a new order.

        :return: Whether the restaurant is enabled and has enough capacity.
        """

        if not self.enabled:
            return False
        if len(self.active_orders) >= self.capacity:
            return False
        return random.random() > 0.01

    def generate_prep_time(self) -> float:
        """
        Generate a lognormal prep time for an order.

        :return: Time in seconds.
        """

        sigma = 0.5
        mu    = np.log(self.avg_prep_time) - (sigma ** 2) / 2
        return random.lognormvariate(mu, sigma)

    def update_preparing_orders_to_ready(self, current_time: float):
        """
        Update status of preparing orders.

        :param current_time: Current simulation time.
        """

        for order in self.active_orders:
            if order.status == 'PREPARING' and current_time >= order.ready_time:
                order.status = 'READY'

    def accept_order(self, order: Order):
        """
        Accept an order from a user.

        :param order: The accepted order.
        """

        self.active_orders.append(order)
        self._sync_enabled_status()

    def remove_order(self, order: Order):
        """
        Remove an order that has been completed or cancelled.

        :param order: The removed order.
        """

        if order in self.active_orders:
            self.active_orders.remove(order)
            self._sync_enabled_status()

    def submit_rating(self, stars: int) -> None:
        """
        Record a 1-5 star rating for the restaurant and update its average.

        :param stars: The number of stars.
        """

        self._rating_sum   += stars
        self._rating_count += 1
        self.rating         = self._rating_sum / self._rating_count

    def _sync_enabled_status(self):
        """
        Synchronize the restaurant's enabled status with its capacity.
        """

        self.enabled = len(self.active_orders) < self.capacity


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class User:
    """
    Static agent. Generates orders to restaurants and gets served by drivers. 
    
    Fully decoupled from Simulaton:
      - Interacts with restaurants by providing ratings.
      - Bases choices of restaurants on other user's ratings 
    """
    def __init__(self, user_id: int, location: int):
        self.user_id  = user_id
        self.location = location


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class Driver:
    """
    Mobile agent. Moves along graph edges using distance-based interpolation.

    Fully decoupled from Simulation:
      - Routes are pushed in via assign_order() / assign_route()
      - update_position() returns DriverEvents instead of calling back into Simulation
      - Simulation observes events and reacts (timestamps, restaurant cleanup, etc.)
    """

    def __init__(self, driver_id: int, location_node: int, speed: float | None = None):
        self.id       = driver_id
        self.location = location_node
        self.coords   = (0.0, 0.0)

        self.current_route:     list[int]   = []
        self.distance_on_edge:  float       = 0.0
        self.current_edge:      tuple[int,int] | None = None

        self.order_queue:      deque[Order] = deque()
        self.active_order:     Order | None = None
        self.status            = 'IDLE'
        self.service_remaining: float       = 0.0
        self.service_type:     str | None   = None
        self.available: bool = True  # set to False to take driver offline after current queue drains

        if speed is None:
            self.speed = max(4.5, random.gauss(6.9, 1.0))
        else:
            self.speed = speed

    # ------------------------------------------------------------------
    # External interface — called by Simulation only
    # ------------------------------------------------------------------

    def assign_order(self, order: Order, pickup_route: list[int]) -> None:
        """
        Push an order and pre-computed pickup route in.

        :param order:  The order to start.
        :param pickup_route: Route to the user
        """
        self.order_queue.append(order)
        if self.status == 'IDLE':
            self._begin_pickup(pickup_route)

    def assign_route(self, route: list[int]) -> None:
        """Push a repositioning route in (used by RepositioningPolicy)."""
        self.set_route(route)

    # ------------------------------------------------------------------
    # Movement engine
    # ------------------------------------------------------------------

    def update_position(self, step_size: float, env: Environment) -> list[tuple]:
        """
        Advances driver by step_size seconds.
        Returns list of (DriverEvent, order) tuples.
        The order reference is always included so Simulation does not need to
        read driver.active_order — which may already be cleared by the time
        the event is processed.
        """
        events: list[tuple] = []

        # --- Service dwell (pickup wait / dropoff handoff) ---
        if self.status in ('PICKUP_SERVICE', 'DROPOFF_SERVICE'):
            self.service_remaining -= step_size
            if self.service_remaining <= 0:
                order = self.active_order          # capture before any clearing
                if order is None: 
                    raise ValueError("No active order")
                if self.service_type == 'PICKUP':
                    order.status = 'PICKED_UP'
                    events.append((DriverEvent.PICKUP_COMPLETE, order))
                    self._begin_delivery()
                elif self.service_type == 'DROPOFF':
                    order.status = 'DELIVERED'
                    events.append((DriverEvent.DROPOFF_COMPLETE, order))
                    self.order_queue.popleft()
                    self.status       = 'IDLE'
                    self.active_order = None
                    events.append((DriverEvent.WENT_IDLE, None))
            return events

        # --- No route ---
        if not self.current_route or len(self.current_route) < 2:
            if (self.status == 'PICKING_UP'
                    and self.active_order
                    and self.active_order.status == 'READY'):
                self._handle_arrival(events)
            return events

        # --- Advance along route ---
        self.distance_on_edge += self.speed * step_size

        while len(self.current_route) >= 2:
            if self.current_edge is None:
                raise ValueError("Current edge is none")
            u, v     = self.current_edge
            edge_len = env.get_edge_data(u, v)['length']

            if self.distance_on_edge < edge_len:
                break

            self.distance_on_edge -= edge_len
            self.current_route.pop(0)

            if len(self.current_route) < 2:
                self.location = v
                self._handle_arrival(events)
                return events

            self.current_edge = (self.current_route[0], self.current_route[1])

        self._update_coords(env)
        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _begin_pickup(self, route: list[int]):
        """
        Move driver to the first node in the pickup route.

        :param route: list of nodes representing the order's pickup route
        :return: None
        """
        self.status       = 'PICKING_UP'
        self.active_order = self.order_queue[0]
        self.set_route(route)

    def _begin_delivery(self):
        """
        Move driver to user location after completing a pickup.

        :return: None
        """
        self.status = 'DELIVERING'
        if self.active_order is not None:
            self.set_route(self.active_order.route_to_user)

    def set_route(self, route_nodes: list[int]):
        """
        Update the current route for this driver.

        :param route_nodes: list of nodes representing the new route
        :return: None
        """
        if not route_nodes or len(route_nodes) < 2:
            self.current_route = []
            self.current_edge  = None
            return
        self.current_route    = list(route_nodes)
        self.current_edge     = (self.current_route[0], self.current_route[1])
        self.distance_on_edge = 0.0
        self.location         = route_nodes[0]

    def _handle_arrival(self, events: list[tuple]):
        """
        Driver has arrived at the current node.

        :param events: list of (event, order) tuples to be processed by Simulation
        :return: None
        """
        if self.status == 'PICKING_UP':
            self.status            = 'PICKUP_SERVICE'
            self.service_remaining = self._gen_pickup_service_time()
            self.service_type      = 'PICKUP'
            events.append((DriverEvent.ARRIVED_AT_RESTAURANT, self.active_order))
        elif self.status == 'DELIVERING':
            self.status            = 'DROPOFF_SERVICE'
            self.service_remaining = self._gen_dropoff_service_time()
            self.service_type      = 'DROPOFF'
            events.append((DriverEvent.ARRIVED_AT_USER, self.active_order))

    def _update_coords(self, env: Environment):
        """
        Update driver location based on current edge and distance.

        :param env: instance of Environment with graph data
        :return: None
        """
        if not self.current_edge:
            return
        u, v      = self.current_edge
        edge_data = env.get_edge_data(u, v)
        edge_len  = edge_data['length']
        lon_u, lat_u = env.get_node_coords(u)
        lon_v, lat_v = env.get_node_coords(v)
        line = edge_data.get('geometry', LineString([(lon_u, lat_u), (lon_v, lat_v)]))
        if edge_len > 0:
            fraction    = self.distance_on_edge / edge_len
            point       = line.interpolate(min(fraction, 1.0), normalized=True)
            self.coords = (point.y, point.x)  # (lat, lon)

    def _gen_pickup_service_time(self) -> float:
        """
        Generate service time for pickup.

        :return: float representing the service time
        """
        # Mean ~3.5 min. Source: Grubhub MDRP instances (Reyes et al. 2018)
        mu = np.log(210) - (0.45 ** 2) / 2
        return random.lognormvariate(mu, 0.45)

    def _gen_dropoff_service_time(self) -> float:
        """Samples dropoff dwell time for a dense apartment context.

        Uses a two-component Gaussian Mixture to reflect bimodal
        delivery behavior observed in high-rise urban settings:
        quick handoffs (customer waiting at door) vs. full building
        access (intercom, elevator, unit search).

        Zona Tec is predominantly mid-rise apartments, so the slow
        component is weighted higher than in mixed residential areas.

        Source: Zheng et al. 2022 (Complex & Intelligent Systems,
        DOI: 10.1007/s40747-022-00719-4). Component means and weights
        adjusted for Monterrey residential context.

        Returns:
            float: Dropoff service time in seconds.
        """
        # 30% quick handoff (~2 min), 70% full building access (~6 min)
        if random.random() < 0.30:
            mu = np.log(120) - (0.35 ** 2) / 2
            return random.lognormvariate(mu, 0.35)
        else:
            mu = np.log(360) - (0.45 ** 2) / 2
            return random.lognormvariate(mu, 0.45)

