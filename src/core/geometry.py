"""
Geometry utilities for track interpolation and boundary computation.

This module handles the mathematical transformation from discrete control points
to smooth, continuous track geometry using spline interpolation.

Key algorithms:
- Catmull-Rom spline interpolation for smooth curves
- Arc length parameterization for uniform sampling
- Normal vector computation for track boundaries
- Curvature calculation for physics models
"""

import numpy as np
from typing import List, Tuple
from .track import Track, TrackSample, ControlPoint


def catmull_rom_spline(points: np.ndarray, t: float) -> np.ndarray:
    """
    Evaluate Catmull-Rom spline at parameter t.
    
    Catmull-Rom splines pass through all control points and provide
    smooth interpolation with continuous first derivatives.
    
    Args:
        points: Array of 4 control points [P0, P1, P2, P3]
        t: Parameter in [0, 1], where 0=P1, 1=P2
    
    Returns:
        Interpolated point between P1 and P2
    """
    # Catmull-Rom basis matrix
    # This creates smooth curves that pass through P1 and P2
    # using P0 and P3 to determine tangent directions
    t2 = t * t
    t3 = t2 * t
    
    # Catmull-Rom coefficients
    c0 = -0.5 * t3 + t2 - 0.5 * t
    c1 = 1.5 * t3 - 2.5 * t2 + 1.0
    c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    c3 = 0.5 * t3 - 0.5 * t2
    
    return c0 * points[0] + c1 * points[1] + c2 * points[2] + c3 * points[3]


def catmull_rom_tangent(points: np.ndarray, t: float) -> np.ndarray:
    """
    Compute tangent vector of Catmull-Rom spline at parameter t.
    
    The tangent vector points in the direction of motion along the curve.
    
    Args:
        points: Array of 4 control points [P0, P1, P2, P3]
        t: Parameter in [0, 1]
    
    Returns:
        Tangent vector (not normalized)
    """
    t2 = t * t
    
    # Derivatives of Catmull-Rom coefficients
    dc0 = -1.5 * t2 + 2.0 * t - 0.5
    dc1 = 4.5 * t2 - 5.0 * t
    dc2 = -4.5 * t2 + 4.0 * t + 0.5
    dc3 = 1.5 * t2 - t
    
    return dc0 * points[0] + dc1 * points[1] + dc2 * points[2] + dc3 * points[3]


def compute_arc_length(points: np.ndarray, num_samples: int = 100) -> float:
    """
    Compute arc length of a spline segment using numerical integration.
    
    Arc length is crucial for uniform sampling - we want points spaced
    evenly by distance, not by parameter value.
    
    Args:
        points: 4 control points defining the spline segment
        num_samples: Number of samples for numerical integration
    
    Returns:
        Arc length of the segment
    """
    length = 0.0
    prev_point = catmull_rom_spline(points, 0.0)
    
    for i in range(1, num_samples + 1):
        t = i / num_samples
        curr_point = catmull_rom_spline(points, t)
        length += np.linalg.norm(curr_point - prev_point)
        prev_point = curr_point
    
    return length


def interpolate_track_centerline(track: Track, target_spacing: float = 1.0) -> None:
    """
    Generate smooth centerline from control points using Catmull-Rom splines.
    
    This is the core function that transforms discrete control points into
    a continuous, smooth track centerline with uniform spacing.
    
    Args:
        track: Track object to populate with samples
        target_spacing: Desired spacing between samples in meters
    """
    if not track.is_valid():
        raise ValueError("Invalid track definition")
    
    control_points = track.get_centerline_points()
    control_widths = track.get_width_at_points()
    n_controls = len(control_points)
    
    if n_controls < 4:
        raise ValueError("Need at least 4 control points for spline interpolation")
    
    track.samples = []
    current_s = 0.0
    
    # For each spline segment between consecutive control points
    for i in range(n_controls - 1 if not track.closed else n_controls):
        # Get 4 points for Catmull-Rom spline (P0, P1, P2, P3)
        # P1 and P2 are the segment endpoints, P0 and P3 provide tangent info
        if track.closed:
            p0 = control_points[(i - 1) % n_controls]
            p1 = control_points[i % n_controls]
            p2 = control_points[(i + 1) % n_controls]
            p3 = control_points[(i + 2) % n_controls]
            
            w1 = control_widths[i % n_controls]
            w2 = control_widths[(i + 1) % n_controls]
        else:
            # Handle open tracks (less common for racing)
            if i == 0:
                p0 = control_points[0] - (control_points[1] - control_points[0])
            else:
                p0 = control_points[i - 1]
            
            p1 = control_points[i]
            p2 = control_points[i + 1]
            
            if i == n_controls - 2:
                p3 = control_points[-1] + (control_points[-1] - control_points[-2])
            else:
                p3 = control_points[i + 2]
            
            w1 = control_widths[i]
            w2 = control_widths[i + 1]
        
        spline_points = np.array([p0, p1, p2, p3])
        
        # Compute arc length of this segment
        segment_length = compute_arc_length(spline_points)
        num_samples = max(2, int(segment_length / target_spacing))
        
        # Generate samples along this segment
        for j in range(num_samples):
            if i > 0 and j == 0:
                continue  # Skip first point of subsequent segments (avoid duplicates)
            
            t = j / num_samples
            
            # Interpolate position
            pos = catmull_rom_spline(spline_points, t)
            
            # Interpolate width
            width = w1 * (1 - t) + w2 * t
            
            # Compute tangent and normalize it
            tangent = catmull_rom_tangent(spline_points, t)
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 1e-8:  # Avoid division by zero
                tangent_unit = tangent / tangent_norm
                
                # Normal vector is tangent rotated 90° counterclockwise
                # This points to the "left" side of the track
                normal_unit = np.array([-tangent_unit[1], tangent_unit[0]])
            else:
                # Fallback for degenerate cases
                tangent_unit = np.array([1.0, 0.0])
                normal_unit = np.array([0.0, 1.0])
            
            # Create track sample
            sample = TrackSample(
                s=current_s,
                x=pos[0],
                y=pos[1],
                width=width,
                tangent_x=tangent_unit[0],
                tangent_y=tangent_unit[1],
                normal_x=normal_unit[0],
                normal_y=normal_unit[1]
            )
            
            track.samples.append(sample)
            
            # Update arc length for next point
            if j < num_samples - 1:
                next_t = (j + 1) / num_samples
                next_pos = catmull_rom_spline(spline_points, next_t)
                current_s += np.linalg.norm(next_pos - pos)
    
    track.total_length = current_s
    print(f"Generated {len(track.samples)} samples, total length: {track.total_length:.1f}m")


def compute_track_boundaries(track: Track) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inner and outer track boundaries from centerline samples.
    
    The boundaries define the legal racing area. Cars must stay between
    these lines to avoid penalties.
    
    Args:
        track: Track with populated centerline samples
    
    Returns:
        Tuple of (inner_boundary, outer_boundary) as Nx2 arrays
    """
    if not track.samples:
        raise ValueError("Track must have centerline samples")
    
    inner_points = []
    outer_points = []
    
    for sample in track.samples:
        # Inner boundary (right side of track, normal points left)
        inner_x = sample.x - sample.normal_x * sample.width / 2
        inner_y = sample.y - sample.normal_y * sample.width / 2
        inner_points.append([inner_x, inner_y])
        
        # Outer boundary (left side of track)
        outer_x = sample.x + sample.normal_x * sample.width / 2
        outer_y = sample.y + sample.normal_y * sample.width / 2
        outer_points.append([outer_x, outer_y])
    
    return np.array(inner_points), np.array(outer_points)


def compute_curvature(track: Track) -> np.ndarray:
    """
    Compute curvature at each centerline sample.
    
    Curvature determines the maximum speed a car can take through each
    section without sliding off the track.
    
    Args:
        track: Track with centerline samples
    
    Returns:
        Array of curvature values (1/radius) at each sample
    """
    if len(track.samples) < 3:
        return np.zeros(len(track.samples))
    
    curvatures = []
    n = len(track.samples)
    
    for i in range(n):
        # Use finite differences to approximate curvature
        # Get three consecutive points
        if track.closed:
            prev_idx = (i - 1) % n
            curr_idx = i
            next_idx = (i + 1) % n
        else:
            prev_idx = max(0, i - 1)
            curr_idx = i
            next_idx = min(n - 1, i + 1)
        
        p_prev = np.array([track.samples[prev_idx].x, track.samples[prev_idx].y])
        p_curr = np.array([track.samples[curr_idx].x, track.samples[curr_idx].y])
        p_next = np.array([track.samples[next_idx].x, track.samples[next_idx].y])
        
        # Compute curvature using the circumcircle formula
        # κ = 2 * |det(v1, v2)| / (|v1| * |v2| * |v1 - v2|)
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        v_diff_norm = np.linalg.norm(v1 - v2)
        
        if v1_norm > 1e-8 and v2_norm > 1e-8 and v_diff_norm > 1e-8:
            # Cross product in 2D (gives signed curvature)
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            curvature = 2 * abs(cross_product) / (v1_norm * v2_norm * v_diff_norm)
        else:
            curvature = 0.0
        
        curvatures.append(curvature)
    
    return np.array(curvatures)


if __name__ == "__main__":
    # Demo: Process the example oval
    from .track import create_simple_oval
    
    print("Creating and processing simple oval...")
    oval = create_simple_oval()
    
    # Generate smooth centerline
    interpolate_track_centerline(oval, target_spacing=2.0)
    
    # Compute boundaries
    inner, outer = compute_track_boundaries(oval)
    print(f"Computed boundaries: {len(inner)} inner points, {len(outer)} outer points")
    
    # Compute curvature
    curvatures = compute_curvature(oval)
    print(f"Curvature range: {curvatures.min():.6f} to {curvatures.max():.6f} (1/m)")
    print(f"Max radius: {1/curvatures.max():.1f}m, Min radius: {1/curvatures.min():.1f}m")
    
    # Show some sample points
    print("\nFirst 5 centerline samples:")
    for i in range(min(5, len(oval.samples))):
        s = oval.samples[i]
        print(f"  s={s.s:.1f}m: ({s.x:.1f}, {s.y:.1f}), width={s.width:.1f}m")
