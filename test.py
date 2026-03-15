from agents import *

# 1. Setup Environment
sim = Simulation(step_size=1,graph=None)

# Initialize 10 restaurants with a capacity of 100
for i in range(10):
    res = Restaurant(
        restaurant_id=i, 
        location=100, # fake node 
        rating=5, 
        capacity=100, 
        avg_prep_time=300,
        service_radius=5000
    )
    sim.add_restaurant(res)

target_res_id = 1
print(f"--- Starting simulation for Restaurant {target_res_id} (Capacity: 100) ---")

# 2 initialize 100 users 
for i in range(100):
    user = User(user_id=i,
                location=100 # fake node 
                )
    sim.add_user(user)

# 3. Inject 102 orders at t=0, 1 per user
accepted_count = 0
rejected_count = 0

for i in range(102):
    if sim.process_user_request(user_id=i, restaurant_id=target_res_id):
        accepted_count += 1
    else:
        rejected_count += 1

print(f"Accepted orders: {accepted_count}")
print(f"Rejected orders: {rejected_count}")

# 3. Advance clock to t=600 (10 minutes later)
print(f"\n--- Advancing clock to t=600 ---")
sim.run_until(600)

# 4. Final state analysis
preparing = [o for o in sim.orders if o.status == "PREPARING"]
ready = [o for o in sim.orders if o.status == "READY"]

print(f"Current Simulation Time: {sim.current_time}s")
print(f"Orders still in preparation: {len(preparing)}")
print(f"Orders ready for pickup: {len(ready)}")

active_ids = sim.get_preparing_orders()
print(f"\nSample of active Order IDs: {active_ids[:10]}")