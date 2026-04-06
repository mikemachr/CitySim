import networkx as nx


class Environment:
    """
    Owns the road network and all routing logic.
    No agents, no business logic — just graph queries.
    All other components receive an Environment instance and call it for spatial info.
    """

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self._route_cache: dict[tuple[int, int], tuple] = {}

    def get_route(self, origin: int, destination: int) -> tuple:
        """Returns (distance_m, path_nodes). Cached. Returns (None, None) if no path."""
        if origin == destination:
            return 0.0, [origin]
        key = (origin, destination)
        if key not in self._route_cache:
            try:
                distance = nx.shortest_path_length(self.graph, origin, destination, weight='length')
                path     = nx.shortest_path(self.graph, origin, destination, weight='length')
                self._route_cache[key] = (distance, path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self._route_cache[key] = (None, None)
        return self._route_cache[key]

    def get_reachable(self, origin: int, cutoff_m: float) -> dict[int, float]:
        """Returns {node: distance_m} for all nodes reachable within cutoff_m."""
        return nx.single_source_dijkstra_path_length(
            self.graph, origin, cutoff=cutoff_m, weight='length'
        )

    def get_edge_data(self, u: int, v: int) -> dict:
        """Returns data dict for edge (u, v), first parallel edge."""
        return self.graph.get_edge_data(u, v)[0]

    def get_node_coords(self, node: int) -> tuple[float, float]:
        """Returns (lon, lat) for a graph node."""
        data = self.graph.nodes[node]
        return data['x'], data['y']
