"""
Physics engine for Formula 1 racing line optimization.

This module handles the realistic vehicle dynamics that determine how fast
a car can go through each section of track while staying within physical limits.

Key physics concepts:
- Friction circle: lateral vs longitudinal acceleration tradeoffs
- Speed limits from curvature: v_max = sqrt(μ * g / κ)
- Forward/backward pass: acceleration and braking constraints
- Lap time integration: sum of time through each segment
"""

import numpy as np
from typing import List, Tuple, Optional
from .track import Track, TrackSample
from .geometry import compute_curvature


class VehicleModel:
    """
    Simplified Formula 1 car physics model.
    
    Uses a "friction circle" approach where lateral and longitudinal
    accelerations are coupled through tire grip limits.
    """
    
    def __init__(self, 
                 mu_friction: float = 1.8,      # Friction coefficient (F1 ~1.8-2.5)
                 max_accel: float = 12.0,       # Max acceleration (m/s²)
                 max_brake: float = 15.0,       # Max braking (m/s²)
                 max_speed: float = 100.0,      # Top speed (m/s ~360 km/h)
                 mass: float = 800.0,           # Car mass (kg)
                 downforce_coeff: float = 0.0): # Downforce coefficient (optional)
        """
        Initialize F1 car model with realistic parameters.
        
        Args:
            mu_friction: Tire-track friction coefficient
            max_accel: Maximum acceleration in m/s²
            max_brake: Maximum braking deceleration in m/s²
            max_speed: Maximum straight-line speed in m/s
            mass: Vehicle mass in kg
            downforce_coeff: Downforce coefficient for speed-dependent grip
        """
        self.mu = mu_friction
        self.a_max = max_accel
        self.a_brake = max_brake
        self.v_max = max_speed
        self.mass = mass
        self.downforce_coeff = downforce_coeff
        self.gravity = 9.81  # m/s²
    
    def get_friction_limit(self, speed: float) -> float:
        """
        Get effective friction coefficient including downforce.
        
        Downforce increases with speed squared, providing more grip
        at high speeds (realistic F1 behavior).
        
        Args:
            speed: Current speed in m/s
        
        Returns:
            Effective friction coefficient
        """
        # Simple downforce model: μ_eff = μ_base + k * v²
        downforce_factor = self.downforce_coeff * speed * speed
        return self.mu + downforce_factor
    
    def max_cornering_speed(self, curvature: float, speed: float = 0.0) -> float:
        """
        Calculate maximum cornering speed for given curvature.
        
        Based on lateral acceleration limit: a_lat = v² * κ ≤ μ * g
        Therefore: v_max = sqrt(μ * g / κ)
        
        Args:
            curvature: Track curvature (1/radius) in 1/m
            speed: Current speed for downforce calculation
        
        Returns:
            Maximum safe cornering speed in m/s
        """
        if curvature < 1e-6:  # Essentially straight
            return self.v_max
        
        mu_eff = self.get_friction_limit(speed)
        v_corner = np.sqrt(mu_eff * self.gravity / curvature)
        
        return min(v_corner, self.v_max)


def compute_speed_profile(track: Track, vehicle: VehicleModel) -> Tuple[np.ndarray, float]:
    """
    Compute realistic speed profile using forward/backward pass algorithm.
    
    This is the classic racing line algorithm:
    1. Compute curvature-limited speeds
    2. Forward pass: respect acceleration limits
    3. Backward pass: respect braking limits
    
    Args:
        track: Track with centerline samples
        vehicle: Vehicle physics model
    
    Returns:
        Tuple of (speed_profile, lap_time) where speeds are in m/s
    """
    if not track.samples:
        raise ValueError("Track must have centerline samples")
    
    n = len(track.samples)
    curvatures = compute_curvature(track)
    
    # Step 1: Compute curvature-limited speeds
    v_limit = np.zeros(n)
    for i in range(n):
        # Use current speed estimate for downforce (iterative refinement possible)
        v_limit[i] = vehicle.max_cornering_speed(curvatures[i], 50.0)  # Assume 50 m/s for downforce calc
    
    # Step 2: Forward pass - respect acceleration limits
    v_forward = np.zeros(n)
    v_forward[0] = v_limit[0]  # Start at curvature limit
    
    for i in range(1, n):
        # Distance to next point
        if i < n - 1:
            ds = track.samples[i].s - track.samples[i-1].s
        else:
            # Last segment for closed track
            if track.closed:
                ds = track.total_length - track.samples[i-1].s + track.samples[0].s
            else:
                ds = track.samples[i].s - track.samples[i-1].s
        
        # Maximum speed achievable with acceleration constraint
        # v² = v₀² + 2*a*s
        v_accel = np.sqrt(v_forward[i-1]**2 + 2 * vehicle.a_max * ds)
        
        # Take minimum of curvature limit and acceleration limit
        v_forward[i] = min(v_limit[i], v_accel)
    
    # Step 3: Backward pass - respect braking limits
    v_profile = np.copy(v_forward)
    
    # Go backwards through the track
    for i in range(n-2, -1, -1):
        # Distance to next point
        if i < n - 1:
            ds = track.samples[i+1].s - track.samples[i].s
        else:
            # Handle wraparound for closed tracks
            if track.closed:
                ds = track.total_length - track.samples[i].s + track.samples[0].s
            else:
                ds = 0  # No next point for open tracks
        
        if ds > 0:
            # Maximum speed that allows braking to next point's speed
            # v₀² = v² - 2*a*s (note: a_brake is positive, so we subtract)
            v_brake = np.sqrt(v_profile[i+1]**2 + 2 * vehicle.a_brake * ds)
            
            # Take minimum of forward pass and braking constraint
            v_profile[i] = min(v_profile[i], v_brake)
    
    # Step 4: Compute lap time
    lap_time = 0.0
    for i in range(n):
        if i < n - 1:
            ds = track.samples[i+1].s - track.samples[i].s
        else:
            if track.closed:
                ds = track.total_length - track.samples[i].s
            else:
                ds = 0
        
        if ds > 0 and v_profile[i] > 0:
            # Average speed for this segment
            if i < n - 1:
                v_avg = (v_profile[i] + v_profile[i+1]) / 2
            else:
                v_avg = (v_profile[i] + v_profile[0]) / 2 if track.closed else v_profile[i]
            
            lap_time += ds / v_avg
    
    return v_profile, lap_time


def analyze_performance(track: Track, speeds: np.ndarray, vehicle: VehicleModel) -> dict:
    """
    Analyze lap performance and identify key sections.
    
    Args:
        track: Track with samples
        speeds: Speed profile from compute_speed_profile
        vehicle: Vehicle model used
    
    Returns:
        Dictionary with performance analysis
    """
    if not track.samples or len(speeds) == 0:
        return {}
    
    curvatures = compute_curvature(track)
    
    # Convert speeds to km/h for readability
    speeds_kmh = speeds * 3.6
    
    # Find interesting sections
    max_speed_idx = np.argmax(speeds)
    min_speed_idx = np.argmin(speeds)
    
    # Find braking zones (large speed drops)
    speed_drops = []
    for i in range(1, len(speeds)):
        speed_drop = speeds[i-1] - speeds[i]
        if speed_drop > 5.0:  # Significant braking (>5 m/s drop)
            speed_drops.append((i, speed_drop))
    
    # Find acceleration zones (large speed gains)
    speed_gains = []
    for i in range(1, len(speeds)):
        speed_gain = speeds[i] - speeds[i-1]
        if speed_gain > 3.0:  # Significant acceleration (>3 m/s gain)
            speed_gains.append((i, speed_gain))
    
    analysis = {
        'max_speed_kmh': float(speeds_kmh[max_speed_idx]),
        'max_speed_location': float(track.samples[max_speed_idx].s),
        'min_speed_kmh': float(speeds_kmh[min_speed_idx]),
        'min_speed_location': float(track.samples[min_speed_idx].s),
        'avg_speed_kmh': float(np.mean(speeds_kmh)),
        'max_curvature': float(np.max(curvatures)),
        'max_curvature_location': float(track.samples[np.argmax(curvatures)].s),
        'num_braking_zones': len(speed_drops),
        'num_acceleration_zones': len(speed_gains),
        'speed_range_kmh': float(speeds_kmh[max_speed_idx] - speeds_kmh[min_speed_idx])
    }
    
    return analysis


if __name__ == "__main__":
    # Demo: Compute realistic lap times for our oval
    from .track import create_simple_oval
    from .geometry import interpolate_track_centerline
    
    print("=== F1 Physics Engine Demo ===")
    
    # Create and process track
    oval = create_simple_oval()
    interpolate_track_centerline(oval, target_spacing=2.0)
    
    # Create realistic F1 car model
    f1_car = VehicleModel(
        mu_friction=1.8,        # F1 slicks on dry asphalt
        max_accel=12.0,         # ~1.2g acceleration
        max_brake=15.0,         # ~1.5g braking
        max_speed=100.0,        # ~360 km/h top speed
        downforce_coeff=0.001   # Small downforce effect
    )
    
    print(f"Vehicle: μ={f1_car.mu}, a_max={f1_car.a_max}m/s², brake_max={f1_car.a_brake}m/s²")
    
    # Compute speed profile
    speeds, lap_time = compute_speed_profile(oval, f1_car)
    
    print(f"\n=== Lap Performance ===")
    print(f"Lap time: {lap_time:.2f} seconds")
    print(f"Average speed: {np.mean(speeds)*3.6:.1f} km/h")
    print(f"Top speed: {np.max(speeds)*3.6:.1f} km/h")
    print(f"Minimum speed: {np.min(speeds)*3.6:.1f} km/h")
    
    # Detailed analysis
    analysis = analyze_performance(oval, speeds, f1_car)
    print(f"\n=== Detailed Analysis ===")
    print(f"Speed range: {analysis['speed_range_kmh']:.1f} km/h")
    print(f"Max speed at s={analysis['max_speed_location']:.1f}m: {analysis['max_speed_kmh']:.1f} km/h")
    print(f"Min speed at s={analysis['min_speed_location']:.1f}m: {analysis['min_speed_kmh']:.1f} km/h")
    print(f"Braking zones: {analysis['num_braking_zones']}")
    print(f"Acceleration zones: {analysis['num_acceleration_zones']}")
    
    # Show speed at key points
    print(f"\n=== Speed Profile Sample ===")
    for i in range(0, len(speeds), len(speeds)//8):  # Show 8 points around track
        s = oval.samples[i]
        speed_kmh = speeds[i] * 3.6
        curvature = compute_curvature(oval)[i]
        radius = 1/curvature if curvature > 1e-6 else float('inf')
        print(f"s={s.s:6.1f}m: {speed_kmh:5.1f} km/h (R={radius:6.1f}m)")
