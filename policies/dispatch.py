from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from typing import TYPE_CHECKING

from policies.base import DispatchPolicy

if TYPE_CHECKING:
    from environment import Environment


class GreedyPolicy(DispatchPolicy):
    """
    FIFO greedy dispatch.
    Pairs the next pending order with any idle driver, no optimisation.
    Intended to be called every tick so there is no batching delay.
    """

    def assign(
        self,
        idle_driver_locations: dict[int, int],
        pending_orders: dict[int, int],
        env: Environment,
    ) -> list[tuple[int, int]]:

        assignments = []

        # Use a local deque to preserve arrival order
        order_queue = deque(pending_orders.keys())
        available   = list(idle_driver_locations.keys())

        while order_queue and available:
            order_id  = order_queue.popleft()
            driver_id = available.pop()
            assignments.append((driver_id, order_id))

        return assignments


class HungarianPolicy(DispatchPolicy):
    """
    Batch bipartite matching via the Hungarian algorithm.
    Minimises total driver-to-restaurant travel distance across all
    idle-driver / unassigned-order pairs reachable within pickup_radius.
    """

    def __init__(self, pickup_radius: float = 3000):
        self.pickup_radius = pickup_radius

    def assign(
        self,
        idle_driver_locations: dict[int, int],
        pending_orders: dict[int, int],
        env: Environment,
    ) -> list[tuple[int, int]]:

        driver_ids = list(idle_driver_locations.keys())
        order_ids  = list(pending_orders.keys())

        if not driver_ids or not order_ids:
            return []

        cost_matrix = np.full((len(driver_ids), len(order_ids)), np.inf)

        restaurant_nodes = [pending_orders[oid] for oid in order_ids]

        for i, driver_id in enumerate(driver_ids):
            #reachable = env.get_reachable(idle_driver_locations[driver_id], self.pickup_radius)
            # use cached 
            reachable = env.get_reachable_cached(idle_driver_locations[driver_id], self.pickup_radius)
            cost_matrix[i] = np.array([
                reachable.get(node, np.inf) for node in restaurant_nodes
            ])

        valid_rows = ~np.all(np.isinf(cost_matrix), axis=1)
        valid_cols = ~np.all(np.isinf(cost_matrix), axis=0)
        filtered   = cost_matrix[np.ix_(valid_rows, valid_cols)]

        if filtered.size == 0:
            return []

        row_ind, col_ind = linear_sum_assignment(filtered)
        row_ind = np.where(valid_rows)[0][row_ind]
        col_ind = np.where(valid_cols)[0][col_ind]

        return [
            (driver_ids[r], order_ids[c])
            for r, c in zip(row_ind, col_ind)
        ]
