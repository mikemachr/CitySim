# test.py (Bridging with real Graph data)

from agents import *
from routing import distrito_tec



# 2. Initialize Simulation with the real graph
sim = Simulation(step_size=1, graph=distrito_tec())

# 3. Setup Entities using Real Node IDs

shared_node = 1433046447 
resitential_node = 1682152103
# Add a restaurant
res = Restaurant(
    restaurant_id=1, 
    location=shared_node, 
    rating=5, 
    capacity=100, 
    avg_prep_time=300,
    service_radius=50000
)
sim.add_restaurant(res)

# Add 100 users at the same residential node 
for i in range(100):
    user = User(user_id=i, location=resitential_node)
    sim.add_user(user)

# 4. Run the process
accepted = 0
for i in range(100):
    if sim.process_user_request(user_id=i, restaurant_id=1):
        accepted += 1

print(f"Simulation tick 0: {accepted} orders accepted at Restaurant 1")

# 5. Advance Time
sim.run_until(600)
ready = [o for o in sim.orders if o.status == "READY"]
print(f"After 10 mins: {len(ready)} orders are ready for pickup.")