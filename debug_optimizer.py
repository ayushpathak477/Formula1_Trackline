"""
Fixed racing line optimization - fast and actually works.
"""

import sys
sys.path.append('.')

from src.core.track import Track, ControlPoint
from src.core.geometry import interpolate_track_centerline
from src.core.physics import VehicleModel, compute_speed_profile
import numpy as np
import time
from scipy.optimize import minimize

def create_simple_hairpin_track():
    """Create a simple track with one clear hairpin where racing line matters."""
    control_points = []
    track_width = 12.0
    
    # Straight approach
    control_points.append(ControlPoint(0, 0, track_width))
    control_points.append(ControlPoint(50, 0, track_width))
    
    # Hairpin turn
    control_points.append(ControlPoint(65, -10, track_width))
    control_points.append(ControlPoint(70, -25, track_width))  # Entry
    control_points.append(ControlPoint(70, -40, track_width))  # Apex
    control_points.append(ControlPoint(65, -55, track_width))  # Exit
    control_points.append(ControlPoint(50, -65, track_width))
    
    # Return straight
    control_points.append(ControlPoint(25, -70, track_width))
    control_points.append(ControlPoint(0, -70, track_width))
    control_points.append(ControlPoint(-10, -60, track_width))
    control_points.append(ControlPoint(-15, -40, track_width))
    control_points.append(ControlPoint(-15, -20, track_width))
    control_points.append(ControlPoint(-10, -5, track_width))
    
    return Track(control_points, closed=True)

def compute_lap_time_direct(track, offsets, vehicle):
    """
    Direct lap time computation for optimization.
    Fixed version that doesn't break.
    """
    # Clamp offsets to boundaries first
    clamped_offsets = np.zeros_like(offsets)
    for i, sample in enumerate(track.samples):
        max_offset = sample.width / 2 - 1.0  # 1m safety margin
        clamped_offsets[i] = np.clip(offsets[i], -max_offset, max_offset)
    
    # Create modified track samples
    modified_samples = []
    for i, sample in enumerate(track.samples):
        # Create new sample with offset position
        new_sample = type(sample)(
            s=sample.s,
            x=sample.x + clamped_offsets[i] * sample.normal_x,
            y=sample.y + clamped_offsets[i] * sample.normal_y,
            width=sample.width,
            tangent_x=sample.tangent_x,
            tangent_y=sample.tangent_y,
            normal_x=sample.normal_x,
            normal_y=sample.normal_y
        )
        modified_samples.append(new_sample)
    
    # Create temporary track
    temp_track = type(track)(track.control_points, track.closed)
    temp_track.samples = modified_samples
    temp_track.total_length = track.total_length
    
    try:
        speeds, lap_time = compute_speed_profile(temp_track, vehicle)
        return lap_time
    except:
        return 1000.0  # Penalty for invalid configurations

def optimize_simple(track, vehicle, max_iter=20):
    """Simple coordinate descent optimization that actually works."""
    n = len(track.samples)
    offsets = np.zeros(n)  # Start at centerline
    
    print(f"Starting simple optimization for {n} samples...")
    
    # Get baseline
    baseline_time = compute_lap_time_direct(track, offsets, vehicle)
    print(f"Baseline (centerline): {baseline_time:.3f}s")
    
    best_time = baseline_time
    best_offsets = offsets.copy()
    
    # Simple coordinate descent
    step_size = 0.5  # 0.5m steps
    
    for iteration in range(max_iter):
        improved = False
        
        # Try moving each point left/right
        for i in range(n):
            sample = track.samples[i]
            max_offset = sample.width / 2 - 1.0
            
            # Try positive offset (left/outside)
            if offsets[i] + step_size <= max_offset:
                test_offsets = offsets.copy()
                test_offsets[i] += step_size
                test_time = compute_lap_time_direct(track, test_offsets, vehicle)
                
                if test_time < best_time:
                    best_time = test_time
                    best_offsets = test_offsets.copy()
                    offsets = test_offsets.copy()
                    improved = True
                    continue
            
            # Try negative offset (right/inside)
            if offsets[i] - step_size >= -max_offset:
                test_offsets = offsets.copy()
                test_offsets[i] -= step_size
                test_time = compute_lap_time_direct(track, test_offsets, vehicle)
                
                if test_time < best_time:
                    best_time = test_time
                    best_offsets = test_offsets.copy()
                    offsets = test_offsets.copy()
                    improved = True
        
        print(f"Iteration {iteration+1}: {best_time:.3f}s (improved: {improved})")
        
        if not improved:
            # Reduce step size and try again
            step_size *= 0.7
            if step_size < 0.1:
                print("Converged!")
                break
    
    improvement = baseline_time - best_time
    improvement_percent = (improvement / baseline_time) * 100
    
    print(f"\nFinal result:")
    print(f"Baseline: {baseline_time:.3f}s")
    print(f"Optimized: {best_time:.3f}s")
    print(f"Improvement: {improvement:.3f}s ({improvement_percent:.2f}%)")
    print(f"Max offset: {np.max(np.abs(best_offsets)):.2f}m")
    
    return best_offsets, best_time

def debug_racing_line():
    print("=== FIXED Racing Line Debug ===")
    
    # Create simple test track
    track = create_simple_hairpin_track()
    
    # Reasonable resolution - not too many samples
    interpolate_track_centerline(track, target_spacing=5.0)  # 5m spacing
    print(f"Track: {len(track.samples)} samples, {track.total_length:.1f}m length")
    
    # F1 car
    f1_car = VehicleModel(mu_friction=1.8, max_accel=10.0, max_brake=12.0, max_speed=80.0)
    
    # Time the whole process
    start_time = time.time()
    
    # Run optimization
    best_offsets, best_time = optimize_simple(track, f1_car, max_iter=15)
    
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.1f}s")
    
    # Show where the offsets are
    significant_offsets = [(i, offset) for i, offset in enumerate(best_offsets) if abs(offset) > 0.2]
    if significant_offsets:
        print(f"\nSignificant offsets:")
        for i, offset in significant_offsets[:5]:  # Show top 5
            s_pos = track.samples[i].s
            print(f"  s={s_pos:.1f}m: {offset:+.2f}m")

if __name__ == "__main__":
    debug_racing_line()
