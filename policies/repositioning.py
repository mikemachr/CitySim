from __future__ import annotations
from typing import TYPE_CHECKING

from policies.base import RepositioningPolicy

if TYPE_CHECKING:
    from environment import Environment


class StaticPolicy(RepositioningPolicy):
    """
    No-op repositioning — drivers stay where they are when idle.
    This is the current behaviour and serves as the baseline to beat.
    """

    def reposition(
        self,
        idle_driver_locations: dict[int, int],
        env: Environment,
        sim_state: dict,
    ) -> dict[int, int]:
        # Return current location unchanged for every idle driver
        return dict(idle_driver_locations)


class RLPolicy(RepositioningPolicy):
    """
    RL-based repositioning policy. Socket is wired; model is not yet implemented.

    The observation vector passed to the model is built from sim_state keys:
        - current_time        : float
        - driver_node         : int   (per-driver, injected by Simulation)
        - pending_count       : int
        - restaurant_nodes    : list[int]

    To train:
        policy = RLPolicy()
        policy.train(sim_factory=lambda: Simulation(...), episodes=1000)

    To run:
        policy = RLPolicy(model_path='checkpoints/run1.pt')
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path:
            self._load(model_path)

    def reposition(
        self,
        idle_driver_locations: dict[int, int],
        env: Environment,
        sim_state: dict,
    ) -> dict[int, int]:
        if self.model is None:
            # Fall back to static until a model is loaded
            return dict(idle_driver_locations)

        targets = {}
        for driver_id, node in idle_driver_locations.items():
            obs = self._build_obs(driver_id, node, sim_state)
            target_node = self.model.predict(obs)
            targets[driver_id] = target_node
        return targets

    def train(self, sim_factory, episodes: int):
        """
        Train the RL agent using sim_factory to produce fresh environments.
        sim_factory: callable() -> Simulation
        """
        raise NotImplementedError("RL training not yet implemented.")

    def _build_obs(self, driver_id: int, node: int, sim_state: dict):
        raise NotImplementedError

    def _load(self, path: str):
        raise NotImplementedError
