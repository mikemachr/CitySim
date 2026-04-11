"""run_experiment.py -- Batch experiment runner for the Distrito Tec delivery simulation.

Each experiment is fully self-contained: it gets its own output directory with
a config snapshot, an orders CSV, a metrics JSON, and an optional MP4 animation.
Experiments can be executed sequentially or in parallel via joblib.

Typical notebook usage::

    from run_experiment import ExperimentConfig, run_experiments
    from policies.dispatch import HungarianPolicy, GreedyPolicy
    from policies.repositioning import StaticPolicy
    from routing import distrito_tec

    sub_graph, routable_restaurants, residential_zones = distrito_tec()

    experiments = [
        ExperimentConfig(
            name="hungarian_balanced",
            n_drivers=215,
            dispatch_policy=HungarianPolicy(pickup_radius=3000),
        ),
        ExperimentConfig(
            name="greedy_undersupplied",
            n_drivers=72,
            dispatch_policy=GreedyPolicy(),
            animate=True,
        ),
    ]

    results = run_experiments(
        experiments,
        sub_graph,
        routable_restaurants,
        residential_zones,
        n_jobs=2,
    )
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import geopandas as gpd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for parallel workers
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from agents import Driver, Order, Restaurant, User
from environment import Environment
from policies.base import DispatchPolicy, RepositioningPolicy
from policies.dispatch import HungarianPolicy
from policies.repositioning import StaticPolicy
from routing import get_closest_place_node_id
from simulation import Simulation, generate_orders, get_order_rate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All knobs for a single simulation run.

    Args:
        name: Unique identifier; becomes the output subdirectory name.
        n_drivers: Total driver pool size (all cohorts combined).
        n_users: Number of user agents spawned at startup.
        start_hour: Simulated wall-clock hour at t=0 (0-23).
        warmup_h: Warmup duration in simulated hours. Orders placed during
            warmup are discarded before recording begins.
        sim_hours: Recording duration in simulated hours (after warmup).
        step_size: Simulation tick size in seconds.
        dispatch_interval: Seconds between batch dispatch invocations.
        dispatch_policy: DispatchPolicy instance to use.
        reposition_policy: RepositioningPolicy instance to use.
        seed: RNG seed for reproducibility.
        animate: If True, renders and saves an MP4 alongside the CSV.
        animation_fps: Frames per second for the output MP4.
        output_dir: Root directory; experiment outputs go in output_dir/name/.
    """

    name:               str
    n_drivers:          int             = 215
    n_users:            int             = 10_000
    start_hour:         float           = 11.0
    warmup_h:           float           = 6.0
    sim_hours:          float           = 24.0
    step_size:          float           = 10.0
    dispatch_interval:  float           = 3.0
    dispatch_policy:    DispatchPolicy  = field(
        default_factory=lambda: HungarianPolicy(pickup_radius=3000)
    )
    reposition_policy:  RepositioningPolicy = field(
        default_factory=StaticPolicy
    )
    seed:               int             = 42
    animate:            bool            = False
    animation_fps:      int             = 30
    output_dir:         str             = "results"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seed_rngs(seed: int) -> None:
    """Seeds Python stdlib random and NumPy with the given seed."""
    random.seed(seed)
    np.random.seed(seed)


def _build_simulation(
    cfg: ExperimentConfig,
    sub_graph,
    routable_restaurants: pd.DataFrame,
    residential_zones: pd.DataFrame,
) -> tuple[Simulation, list[int]]:
    """Constructs and wires up a fresh Simulation from a config.

    Returns:
        Tuple of (sim, residential_nodes) where residential_nodes is the list
        of graph node IDs used for driver spawn positions.
    """
    env = Environment(sub_graph)

    sim = Simulation(
        env=env,
        dispatch_policy=cfg.dispatch_policy,
        repositioning_policy=cfg.reposition_policy,
        step_size=cfg.step_size,
        dispatch_interval=cfg.dispatch_interval,
        start_hour=cfg.start_hour - cfg.warmup_h,  # offset so wall clock at end of warmup == start_hour
    )

    # Restaurants
    for i in range(len(routable_restaurants)):
        res_node = get_closest_place_node_id(
            gpd.GeoDataFrame(routable_restaurants.iloc[[i]]), sub_graph
        )
        sim.add_restaurant(Restaurant(
            restaurant_id=i,
            location=res_node,
            rating=round(random.uniform(3.0, 5.0), 1),
            capacity=10,
            avg_prep_time=780,
            service_radius=5000,
        ))

    # Residential spawn nodes -- sample with replacement, one per user
    sampled_zones = residential_zones.sample(cfg.n_users, replace=True)
    residential_nodes = [
        get_closest_place_node_id(gpd.GeoDataFrame(sampled_zones.iloc[[i]]), sub_graph)
        for i in range(cfg.n_users)
    ]

    # Users
    for uid in range(cfg.n_users):
        sim.add_user(User(user_id=uid, location=residential_nodes[uid]))

    # Drivers -- spawn on a random subset of the same residential nodes
    for did in range(cfg.n_drivers):
        location = random.choice(residential_nodes)
        sim.add_driver(Driver(driver_id=did, location_node=location))

    return sim, residential_nodes


def _schedule_driver_shifts(sim: Simulation, residential_nodes: list[int]) -> None:
    """Applies the three-cohort shift schedule (morning / evening / always-on).

    Shift structure (simulated wall clock):
        Morning shift : 09:00 - 15:00  (~30 % of fleet)
        Evening shift : 17:00 - 23:00  (~50 % of fleet)
        Always-on     :                (~20 % of fleet)

    Args:
        sim: Fully populated Simulation instance.
        residential_nodes: Pool of valid spawn nodes for repositioning.
    """
    drivers        = list(sim.drivers.values())
    n              = len(drivers)
    morning_cohort = drivers[:int(n * 0.30)]
    evening_cohort = drivers[int(n * 0.30):int(n * 0.80)]

    for d in morning_cohort:
        jitter  = random.uniform(-900, 900)
        start_s = max(0.0, (9.0  - sim.start_hour) * 3600 + jitter)
        end_s   = max(0.0, (15.0 - sim.start_hour) * 3600 + jitter)
        if start_s > 0:
            sim.schedule_event(start_s, "enable_driver",  d.id)
        sim.schedule_event(end_s,   "disable_driver", d.id)

    for d in evening_cohort:
        jitter  = random.uniform(-900, 900)
        start_s = max(0.0, (17.0 - sim.start_hour) * 3600 + jitter)
        end_s   = max(0.0, (23.0 - sim.start_hour) * 3600 + jitter)
        if start_s > 0:
            sim.schedule_event(start_s, "enable_driver",  d.id)
        sim.schedule_event(end_s,   "disable_driver", d.id)


def _run_warmup(sim: Simulation, cfg: ExperimentConfig) -> None:
    """Runs the warmup period and discards orders generated during it.

    Args:
        sim: Simulation instance post-shift scheduling.
        cfg: Experiment configuration (provides warmup_h and step_size).
    """
    warmup_end = cfg.warmup_h * 3600

    def _warmup_tick():
        sim.run_tick()
        generate_orders(sim, get_order_rate(sim))

    while sim.current_time < warmup_end:
        _warmup_tick()

    # Discard warmup orders; preserve agent positions and shift schedule
    for oid in list(sim.orders.keys()):
        sim.orders.pop(oid)
    sim.order_id_counter = 1
    sim.pending_orders.clear()
    sim._pending_set.clear()
    sim._active_user_ids.clear()


def _run_recording(sim: Simulation, cfg: ExperimentConfig) -> None:
    """Runs the recording phase tick-by-tick without animation.

    Args:
        sim: Simulation instance post-warmup.
        cfg: Experiment configuration (provides sim_hours and step_size).
    """
    end_time = sim.current_time + cfg.sim_hours * 3600
    while sim.current_time < end_time:
        sim.run_tick()
        generate_orders(sim, get_order_rate(sim))


def _run_recording_animated(
    sim: Simulation,
    cfg: ExperimentConfig,
    out_dir: Path,
) -> None:
    """Runs the recording phase with FuncAnimation driving the sim tick-by-tick.

    Mirrors the playground notebook pattern: one sim tick per frame,
    so driver movement is smooth with no teleporting.

    Args:
        sim: Simulation instance post-warmup.
        cfg: Experiment configuration.
        out_dir: Directory where the MP4 will be written.
    """
    from matplotlib.gridspec import GridSpec

    video_path   = str(out_dir / f"{cfg.name}.mp4")
    drivers_list = list(sim.drivers.values())
    n_frames     = int(cfg.sim_hours * 3600 / cfg.step_size)
    colors       = cm.get_cmap("rainbow")(np.linspace(0, 1, len(drivers_list)))

    graph = sim.env.graph
    fig = plt.figure(figsize=(10, 11), facecolor="black")
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[10, 1], hspace=0.02)
    ax_map = fig.add_subplot(gs[0])
    ax_hud = fig.add_subplot(gs[1])

    for u, v in graph.edges():
        x0, y0 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x1, y1 = graph.nodes[v]["x"], graph.nodes[v]["y"]
        ax_map.plot([x0, x1], [y0, y1], color="#333333", linewidth=0.5, zorder=1)
    node_lons = [d["x"] for _, d in graph.nodes(data=True)]
    node_lats = [d["y"] for _, d in graph.nodes(data=True)]
    ax_map.set_xlim(min(node_lons) - 0.001, max(node_lons) + 0.001)
    ax_map.set_ylim(min(node_lats) - 0.001, max(node_lats) + 0.001)
    ax_map.set_facecolor("black")
    ax_map.axis("off")

    restaurant_lons = [graph.nodes[r.location]["x"] for r in sim.restaurants.values()]
    restaurant_lats = [graph.nodes[r.location]["y"] for r in sim.restaurants.values()]
    ax_map.scatter(
        restaurant_lons, restaurant_lats,
        s=120, c="red", marker="^",
        edgecolors="white", linewidth=1.2, zorder=12,
    )

    dots = ax_map.scatter([], [], c="white", s=60, zorder=10, edgecolors="white", linewidth=0.8)

    ax_hud.set_facecolor("black")
    ax_hud.axis("off")
    hud_text = ax_hud.text(
        0.01, 0.95, "",
        transform=ax_hud.transAxes,
        color="white", fontsize=10, fontweight="bold",
        verticalalignment="top", fontfamily="monospace",
    )

    def _update(frame):
        generate_orders(sim, get_order_rate(sim))
        sim.run_tick()

        current_lons, current_lats, driver_colors = [], [], []
        for i, driver in enumerate(drivers_list):
            if driver.coords != (0.0, 0.0):
                lon, lat = driver.coords[1], driver.coords[0]
            else:
                lon, lat = sim.env.get_node_coords(driver.location)
            current_lons.append(lon)
            current_lats.append(lat)
            if not driver.available:
                driver_colors.append("gray")
            elif driver.status == "IDLE":
                driver_colors.append("white")
            else:
                driver_colors.append(colors[i])

        dots.set_offsets(np.c_[current_lons, current_lats])
        dots.set_color(driver_colors)

        m = sim.metrics_snapshot()
        s = m["orders_by_status"]
        hud_text.set_text(
            f"t={int(sim.current_time)}s  [{m['dispatch_policy']} / {m['repositioning_policy']}]\n"
            f"{sim.wall_clock_display}\n"
            f"PREP:{s['PREPARING']}  READY:{s['READY']}  "
            f"PICKUP:{s['PICKED_UP']}  DONE:{s['DELIVERED']}\n"
            f"Idle:{m['idle_drivers']}  Active:{m['active_drivers']}  "
            f"Off:{m['deactivated_drivers']}\n"
            f"avg e2e:{int(m['avg_end_to_end_s'] or 0)}s"
        )
        return [dots, hud_text]

    ani = FuncAnimation(fig, _update, frames=n_frames, interval=50, blit=True, repeat=False)
    ani.save(video_path, writer="ffmpeg", fps=cfg.animation_fps)
    plt.close(fig)
    logger.info("[%s] Animation saved -> %s", cfg.name, video_path)

def _collect_orders_df(sim: Simulation) -> pd.DataFrame:
    """Builds the per-order ledger DataFrame from sim.orders.

    Args:
        sim: Completed Simulation instance.

    Returns:
        DataFrame with one row per order and all timing columns.
    """
    records = [
        {
            "order_id":          o.id,
            "user_id":           o.user_id,
            "driver_id":         o.driver_id,
            "restaurant_id":     o.restaurant_id,
            "start_time":        o.start_time,
            "assigned_time":     o.assigned_time,
            "pickup_time":       o.pickup_time,
            "delivered_time":    o.delivered_time,
            "end_to_end_s":      o.end_to_end_time,
            "food_wait_s":       o.food_wait_time,
            "dispatch_delay_s":  o.dispatch_delay,
            "prep_time":         o.prep_time,
            "status":            o.status,
            "order_rating":      o.rating,
        }
        for o in sim.orders.values()
    ]
    return pd.DataFrame(records)


def _serialize_config(cfg: ExperimentConfig) -> dict[str, Any]:
    """Converts a config to a JSON-serializable dict.

    Policy objects are represented by their class name since they are not
    JSON-serializable by default.

    Args:
        cfg: Experiment configuration.

    Returns:
        Dict suitable for json.dump().
    """
    d = asdict(cfg)
    d["dispatch_policy"]   = type(cfg.dispatch_policy).__name__
    d["reposition_policy"] = type(cfg.reposition_policy).__name__
    return d


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: ExperimentConfig,
    sub_graph,
    routable_restaurants: pd.DataFrame,
    residential_zones: pd.DataFrame,
) -> dict[str, Any]:
    """Runs a single experiment end-to-end and writes all outputs.

    This function is self-contained and stateless with respect to other
    experiments; it is safe to call from a parallel worker.

    Args:
        cfg: Experiment configuration.
        sub_graph: OSM road network graph (read-only; shared across workers).
        routable_restaurants: GeoDataFrame of restaurant nodes.
        residential_zones: GeoDataFrame of residential polygon centroids.

    Returns:
        Dict containing the final metrics snapshot plus 'name', 'duration_s',
        and 'output_dir' keys for aggregation in the notebook.
    """
    t_start = time.perf_counter()

    # Output directory
    out_dir = Path(cfg.output_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config immediately so the run is auditable even if it crashes
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(_serialize_config(cfg), indent=2))

    _seed_rngs(cfg.seed)

    logger.info("[%s] Building simulation ...", cfg.name)
    sim, residential_nodes = _build_simulation(cfg, sub_graph, routable_restaurants, residential_zones)

    # Disable morning/evening cohorts at t=0; schedule shift windows
    n = cfg.n_drivers
    for i, d in enumerate(sim.drivers.values()):
        if i < int(n * 0.80):
            d.available = False
            sim.idle_drivers.discard(d.id)
    _schedule_driver_shifts(sim, residential_nodes)

    logger.info("[%s] Warmup (%.0f h) ...", cfg.name, cfg.warmup_h)
    _run_warmup(sim, cfg)

    logger.info("[%s] Recording (%.0f h) ...", cfg.name, cfg.sim_hours)
    if cfg.animate:
        _run_recording_animated(sim, cfg, out_dir)
    else:
        _run_recording(sim, cfg)

    # Collect and persist outputs
    df = _collect_orders_df(sim)
    orders_path = out_dir / "orders.csv"
    df.to_csv(orders_path, index=False)

    metrics = sim.metrics_snapshot()
    metrics["name"]       = cfg.name
    metrics["duration_s"] = round(time.perf_counter() - t_start, 1)
    metrics["output_dir"] = str(out_dir)

    metrics_path = out_dir / "metrics.json"
    # metrics_snapshot may contain non-serializable values (e.g. nested dicts)
    metrics_path.write_text(json.dumps(
        {k: v for k, v in metrics.items() if isinstance(v, (str, int, float, bool, type(None)))},
        indent=2,
    ))

    logger.info(
        "[%s] Done in %.1f s -- %d orders, %d delivered.",
        cfg.name, metrics["duration_s"],
        metrics.get("total_orders", 0), metrics.get("n_delivered", 0),
    )
    return metrics


# ---------------------------------------------------------------------------
# Parallel batch launcher
# ---------------------------------------------------------------------------

def run_experiments(
    experiments: list[ExperimentConfig],
    sub_graph,
    routable_restaurants: pd.DataFrame,
    residential_zones: pd.DataFrame,
    n_jobs: int = 1,
    verbose: int = 10,
) -> list[dict[str, Any]]:
    """Runs a list of experiments, optionally in parallel.

    The road network objects are passed by reference and are read-only inside
    each worker, so sharing them across processes is safe for networkx graphs.

    Args:
        experiments: List of ExperimentConfig instances to run.
        sub_graph: OSM road network graph shared across all workers.
        routable_restaurants: GeoDataFrame of restaurant nodes.
        residential_zones: GeoDataFrame of residential polygon centroids.
        n_jobs: Number of parallel workers. 1 = sequential (easiest to debug).
            -1 = use all available cores. Values > 1 use joblib loky backend.
        verbose: joblib verbosity level (0 = silent, 10 = progress per job).

    Returns:
        List of metrics dicts in the same order as the input experiments list.
    """
    if n_jobs == 1:
        return [
            run_experiment(cfg, sub_graph, routable_restaurants, residential_zones)
            for cfg in experiments
        ]

    try:
        from joblib import Parallel, delayed
    except ImportError as exc:
        raise ImportError(
            "joblib is required for parallel execution. "
            "Install it with: pip install joblib"
        ) from exc

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_experiment)(cfg, sub_graph, routable_restaurants, residential_zones)
        for cfg in experiments
    )
    return list(results)
