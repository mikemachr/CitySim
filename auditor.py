import pandas as pd

def detect_incoherent_schedules(df):
    """
    Analyzes a delivery ledger to find drivers with logically impossible schedules.
    
    Checks for:
    1. Chronological violations within an order (e.g., delivered before pickup).
    2. Overlapping deliveries (e.g., picking up Order B before Order A is delivered).
    3. Premature assignments (e.g., assigned before the order was created).
    """
    # Remove rows with missing critical timing data or unassigned drivers
    clean_df = df.dropna(subset=['driver_id', 'assigned_time', 'pickup_time', 'delivered_time']).copy()
    
    incoherences = []

    # Group by driver to analyze their individual timeline
    for driver_id, group in clean_df.groupby('driver_id'):
        # Sort by pickup time to establish the chronological sequence of work
        group = group.sort_values('pickup_time')
        
        # 1. Check Internal Order Logic
        # assigned <= pickup <= delivered AND start <= assigned
        internal_errors = group[
            (group['start_time'] > group['assigned_time']) |
            (group['assigned_time'] > group['pickup_time']) |
            (group['pickup_time'] > group['delivered_time'])
        ]
        
        if not internal_errors.empty:
            incoherences.append({
                'driver_id': driver_id,
                'type': 'Internal Timing Violation',
                'details': internal_errors[['order_id', 'start_time', 'assigned_time', 'pickup_time', 'delivered_time']].to_dict('records')
            })
            continue

        # 2. Check Sequential Logic (Overlap)
        # Does the driver pick up the next order before delivering the previous one?
        # Compare current pickup_time with the previous row's delivered_time
        previous_delivery = group['delivered_time'].shift(1)
        overlaps = group[group['pickup_time'] < previous_delivery]
        
        if not overlaps.empty:
            incoherences.append({
                'driver_id': driver_id,
                'type': 'Overlapping Schedule Violation',
                'details': overlaps[['order_id', 'pickup_time']].assign(prev_delivery=previous_delivery).to_dict('records')
            })

        # Check delivery before prep time elapsed
        prep_violations = group[
            group['delivered_time'] - group['start_time'] < group['prep_time']
        ]
        if not prep_violations.empty:
            incoherences.append({
                'driver_id': driver_id,
                'type': 'Delivered Before Prep Complete',
                'details': prep_violations[['order_id', 'start_time', 'delivered_time', 'prep_time']].to_dict('records')
            })

    return incoherences

# Usage example with your data:
df = pd.read_csv('results.csv')
results = detect_incoherent_schedules(df)
print(f"Found {len(results)} incoherent driver schedules.")