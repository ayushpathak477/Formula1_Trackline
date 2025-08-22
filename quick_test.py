"""
Quick test of racing line optimization with simplified parameters.
"""

import sys
sys.path.append('.')

from src.core.track import create_simple_oval
from src.core.geometry import interpolate_track_centerline
from src.core.physics import VehicleModel, compute_speed_profile
from src.optimization.racing_line import RacingLineOptimizer
import numpy as np

def quick_test():
    print("=== Quick Racing Line Test ===")
    
    # Create simpler track (coarser resolution)
    oval = create_simple_oval(width=100, height=50, track_width=12.0)
    interpolate_track_centerline(oval, target_spacing=5.0)  # Much coarser: 5m spacing
    
    print(f"Track samples: {len(oval.samples)} (total length: {oval.total_length:.1f}m)")
    
    # Simple vehicle model
    car = VehicleModel(mu_friction=1.5, max_accel=10.0, max_brake=12.0, max_speed=80.0)
    
    # Baseline centerline time
    centerline_speeds, centerline_time = compute_speed_profile(oval, car)
    print(f"Centerline lap time: {centerline_time:.2f}s")
    
    # Quick optimization (fewer iterations)
    optimizer = RacingLineOptimizer(car, smoothness_weight=0.1, curvature_weight=0.01)
    
    print("Starting optimization...")
    racing_line = optimizer.optimize(oval, max_iterations=20)  # Much fewer iterations
    
    # Results
    improvement = centerline_time - racing_line.lap_time
    improvement_percent = (improvement / centerline_time) * 100
    
    print(f"\n=== Results ===")
    print(f"Centerline: {centerline_time:.2f}s")
    print(f"Optimal line: {racing_line.lap_time:.2f}s")
    print(f"Improvement: {improvement:.2f}s ({improvement_percent:.1f}%)")
    
    # Show some offsets
    print(f"\nMax offset: {np.max(np.abs(racing_line.offsets)):.1f}m")
    print(f"Offset range: {np.min(racing_line.offsets):.1f} to {np.max(racing_line.offsets):.1f}m")

if __name__ == "__main__":
    quick_test()
