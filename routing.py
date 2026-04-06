import osmnx as ox
import networkx as nx
from copy import deepcopy
import geopandas as gpd

def distrito_tec():
    # 1. Load the real graph (as you did in liltest.ipynb)
    # For testing speed, we can use the Distrito Tec bounding box from your notebook
    place_name = "Monterrey, Nuevo León, Mexico"
    graph = ox.graph_from_place(place_name, network_type="drive")
    north, south, east, west = 25.660, 25.640, -100.280, -100.300
    distrito_tec = (west,south,east,north)

    sub_graph = ox.truncate.truncate_graph_bbox(graph, distrito_tec)

    # Find the largest strongly connected component (where every node can reach every other node)
    largest_scc = max(nx.strongly_connected_components(sub_graph), key=len)
    # Define realistic average speeds for Monterrey urban driving (in km/h)
    hwy_speeds = {
        "residential": 20,
        "tertiary": 25,
        "secondary": 35,
        "primary": 45,
        "unclassified": 15
    }
    # Apply these conservative speeds
    sub_graph = ox.add_edge_speeds(sub_graph, hwy_speeds=hwy_speeds)
    sub_graph = ox.add_edge_travel_times(sub_graph)
    # Create a new graph containing only that component
    sub_graph_connected = deepcopy(sub_graph.subgraph(largest_scc))



    

    graph_area = ox.graph_to_gdfs(sub_graph_connected, nodes=True, edges=False).unary_union.convex_hull
    tags = {
        "amenity": [
            "restaurant",
            "fast_food",
            "cafe",
            "food_court",
            "bar",
            "pub"
        ],
        "shop": [
            "bakery",
            "confectionery"
        ]
    }

    restaurants = ox.features_from_polygon(graph_area, tags=tags)

    # 3. Filter: Only keep restaurants that are close to a valid road node
    def is_routable(geom):
        # Find the nearest node in the SCC
        nearest_node = ox.nearest_nodes(sub_graph_connected, X=geom.centroid.x, Y=geom.centroid.y)
        
        # Calculate distance between restaurant and that road node (in degrees, roughly)
        # If the nearest road is too far (e.g., > 100 meters), it's probably not reachable by car
        node_data = sub_graph_connected.nodes[nearest_node]
        dist = ox.distance.euclidean(geom.centroid.y, geom.centroid.x, node_data['y'], node_data['x'])
        # 0.001 degrees is roughly 100 meters
        return nearest_node if dist < 0.001 else None

    # Apply the filter
    restaurants['nn'] = restaurants.geometry.apply(is_routable)
    routable_restaurants = restaurants.dropna(subset=['nn']).copy()

    # To get residential zones (neighborhood polygons)
    tags = {'landuse': 'residential'}

    # Fetch the data using the existing bounding box
    residential_zones = ox.features_from_bbox(distrito_tec, tags)

    # Alternatively, for individual residential buildings:
    # tags = {'building': ['apartments', 'house', 'residential']}
    # residential_buildings = ox.features_from_bbox(north, south, east, west, tags)
    residential_zones.reset_index(inplace=True)
    routable_restaurants.reset_index(inplace=True)

    return sub_graph_connected,routable_restaurants,residential_zones





def get_closest_place_node_id(place: gpd.GeoDataFrame, G: nx.MultiDiGraph) -> int:
    # 1. Project to a meter-based system (UTM 14N for Monterrey)
    # 2. Calculate centroid in meters
    # 3. Convert that centroid back to degrees (EPSG 4326)
    projected_centroids = place.to_crs(epsg=32614).geometry.centroid
    centroids_in_degrees = projected_centroids.to_crs(epsg=4326)
    
    if len(centroids_in_degrees) == 0:
        raise KeyError("No matches")
        
    # Get the coordinates of the first match
    point = centroids_in_degrees.iloc[0]
    x, y = point.x, point.y
    
    return ox.nearest_nodes(G, X=x, Y=y)