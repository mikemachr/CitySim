from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import Environment


class DispatchPolicy(ABC):
    """
    Decides which idle driver should handle which unassigned order.

    Receives plain dicts so it has zero coupling to Agent internals.
    Returns a list of (driver_id, order_id) pairs to assign.
    """

    @abstractmethod
    def assign(
        self,
        idle_driver_locations: dict[int, int],  # {driver_id: current_node}
        pending_orders: dict[int, int],          # {order_id: restaurant_node}
        env: Environment,
    ) -> list[tuple[int, int]]:                  # [(driver_id, order_id), ...]
        ...


class RepositioningPolicy(ABC):
    """
    Decides where an idle driver should move while waiting for an order.

    sim_state is a plain dict passed by the Simulation — policies can
    read whatever keys they need without coupling to the Simulation class.
    Useful keys: 'current_time', 'pending_count', 'restaurant_nodes'.
    """

    @abstractmethod
    def reposition(
        self,
        idle_driver_locations: dict[int, int],  # {driver_id: current_node}
        env: Environment,
        sim_state: dict,
    ) -> dict[int, int]:                         # {driver_id: target_node}
        ...
