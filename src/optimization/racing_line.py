"""
Racing line optimization engine.

This module implements the core algorithm that finds the optimal racing line
by optimizing lateral offsets from the centerline to minimize lap time.

The optimization problem:
- Variables: lateral offsets d_i at each track sample
- Objective: minimize lap_time(d) + regularization_terms(d)
- Constraints: keep d_i within track boundaries

Key algorithms:
- Gradient-based optimization for smooth, fast convergence
- Regularization to prevent zigzag lines
- Path reconstruction from lateral offsets
- Iterative refinement with physics feedback
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from scipy.optimize import minimize
import copy

from ..core.track import Track, TrackSample
from ..core.geometry import interpolate_track_centerline, compute_curvature
from ..core.physics import VehicleModel, compute_speed_profile


class RacingLine:
    """
    Represents an optimized racing line with performance data.
    """
    
    def __init__(self, track: Track, lateral_offsets: np.ndarray, 
                 speeds: np.ndarray, lap_time: float):
        """
        Initialize racing line result.
        
        Args:
            track: Base track geometry
            lateral_offsets: Lateral offset from centerline at each sample
            speeds: Speed profile along the racing line
            lap_time: Total lap time in seconds
        """
        self.track = track
        self.offsets = lateral_offsets
        self.speeds = speeds
        self.lap_time = lap_time
        
        # Compute racing line coordinates
        self.line_points = self._compute_line_coordinates()
    
    def _compute_line_coordinates(self) -> np.ndarray:
        """Compute actual racing line coordinates from centerline + offsets."""
        line_points = []
        
        for i, sample in enumerate(self.track.samples):
            offset = self.offsets[i]
            
            # Move laterally from centerline using normal vector
            # Positive offset = move left (outside), negative = move right (inside)
            line_x = sample.x + offset * sample.normal_x
            line_y = sample.y + offset * sample.normal_y
            
            line_points.append([line_x, line_y])
        
        return np.array(line_points)
    
    def get_performance_summary(self) -> dict:
        """Get summary of racing line performance."""
        return {
            'lap_time': self.lap_time,
            'avg_speed_kmh': np.mean(self.speeds) * 3.6,
            'max_speed_kmh': np.max(self.speeds) * 3.6,
            'min_speed_kmh': np.min(self.speeds) * 3.6,
            'max_offset': np.max(np.abs(self.offsets)),
            'line_smoothness': np.std(np.diff(self.offsets))  # Lower = smoother
        }


class RacingLineOptimizer:
    """
    Optimizes racing line using gradient-based methods.
    
    The optimizer searches for lateral offsets that minimize lap time
    while maintaining smooth, physically realistic racing lines.
    """
    
    def __init__(self, vehicle: VehicleModel, 
                 smoothness_weight: float = 1.0,
                 curvature_weight: float = 0.1):
        """
        Initialize racing line optimizer.
        
        Args:
            vehicle: Vehicle physics model
            smoothness_weight: Weight for smoothness regularization
            curvature_weight: Weight for curvature regularization
        """
        self.vehicle = vehicle
        self.smoothness_weight = smoothness_weight
        self.curvature_weight = curvature_weight
        
        # Optimization state
        self.track: Optional[Track] = None
        self.best_result: Optional[RacingLine] = None
        self.optimization_history: List[float] = []
    
    def _create_offset_track(self, base_track: Track, offsets: np.ndarray) -> Track:
        """
        Create new track with racing line as centerline.
        
        This allows us to reuse the physics engine by treating the
        racing line as a new "centerline" for speed calculations.
        
        Args:
            base_track: Original track geometry
            offsets: Lateral offsets from original centerline
        
        Returns:
            New track with racing line as centerline
        """
        racing_track = copy.deepcopy(base_track)
        
        # Update sample positions to racing line
        for i, sample in enumerate(racing_track.samples):
            offset = offsets[i]
            
            # Move sample position laterally
            sample.x += offset * sample.normal_x
            sample.y += offset * sample.normal_y
            
            # Note: We keep the same tangent/normal vectors
            # This is an approximation - for exact optimization,
            # we'd recompute geometry, but this is much faster
        
        return racing_track
    
    def _compute_lap_time(self, offsets: np.ndarray) -> float:
        """
        Compute lap time for given lateral offsets.
        
        This is the core objective function that the optimizer minimizes.
        
        Args:
            offsets: Lateral offset at each track sample
        
        Returns:
            Lap time in seconds
        """
        # Clamp offsets to track boundaries
        clamped_offsets = self._clamp_to_boundaries(offsets)
        
        # Create racing line track
        racing_track = self._create_offset_track(self.track, clamped_offsets)
        
        # Compute speed profile and lap time
        try:
            speeds, lap_time = compute_speed_profile(racing_track, self.vehicle)
            return lap_time
        except Exception:
            # Return penalty for invalid configurations
            return 1000.0
    
    def _clamp_to_boundaries(self, offsets: np.ndarray) -> np.ndarray:
        """
        Clamp lateral offsets to stay within track boundaries.
        
        Args:
            offsets: Raw lateral offsets
        
        Returns:
            Clamped offsets within track limits
        """
        clamped = np.copy(offsets)
        
        for i, sample in enumerate(self.track.samples):
            # Track width limits: -width/2 to +width/2
            max_offset = sample.width / 2 - 0.5  # 0.5m safety margin
            clamped[i] = np.clip(offsets[i], -max_offset, max_offset)
        
        return clamped
    
    def _objective_function(self, offsets: np.ndarray) -> float:
        """
        Complete objective function including regularization terms.
        
        The objective balances lap time minimization with line smoothness.
        
        Args:
            offsets: Lateral offsets to evaluate
        
        Returns:
            Total objective value to minimize
        """
        # Primary objective: lap time
        lap_time = self._compute_lap_time(offsets)
        
        # Regularization terms
        smoothness_penalty = 0.0
        curvature_penalty = 0.0
        
        if len(offsets) > 1:
            # Smoothness: penalize large changes in offset
            offset_changes = np.diff(offsets)
            smoothness_penalty = self.smoothness_weight * np.sum(offset_changes**2)
            
            # Curvature: penalize rapid direction changes
            if len(offsets) > 2:
                second_derivatives = np.diff(offset_changes)
                curvature_penalty = self.curvature_weight * np.sum(second_derivatives**2)
        
        total_objective = lap_time + smoothness_penalty + curvature_penalty
        
        # Track optimization progress
        self.optimization_history.append(total_objective)
        
        return total_objective
    
    def optimize(self, track: Track, max_iterations: int = 100, 
                 initial_offsets: Optional[np.ndarray] = None) -> RacingLine:
        """
        Find optimal racing line for the given track.
        
        Uses gradient-based optimization to minimize lap time while
        maintaining smooth, realistic racing lines.
        
        Args:
            track: Track to optimize
            max_iterations: Maximum optimization iterations
            initial_offsets: Starting guess (default: centerline)
        
        Returns:
            Optimized racing line result
        """
        self.track = track
        self.optimization_history = []
        
        if not track.samples:
            raise ValueError("Track must have generated samples")
        
        n_samples = len(track.samples)
        
        # Initial guess: start with centerline (all offsets = 0)
        if initial_offsets is None:
            x0 = np.zeros(n_samples)
        else:
            x0 = np.copy(initial_offsets)
        
        print(f"Optimizing racing line for {n_samples} samples...")
        print(f"Initial lap time: {self._compute_lap_time(x0):.2f}s")
        
        # Set up optimization bounds (track boundaries)
        bounds = []
        for sample in track.samples:
            max_offset = sample.width / 2 - 0.5  # 0.5m safety margin
            bounds.append((-max_offset, max_offset))
        
        # Run optimization
        result = minimize(
            fun=self._objective_function,
            x0=x0,
            method='L-BFGS-B',  # Good for bound-constrained problems
            bounds=bounds,
            options={
                'maxiter': max_iterations,
                'disp': True,
                'ftol': 1e-6  # Convergence tolerance
            }
        )
        
        print(f"Optimization completed in {result.nit} iterations")
        print(f"Final objective: {result.fun:.6f}")
        
        # Compute final racing line
        optimal_offsets = result.x
        racing_track = self._create_offset_track(track, optimal_offsets)
        speeds, lap_time = compute_speed_profile(racing_track, self.vehicle)
        
        # Create result
        racing_line = RacingLine(track, optimal_offsets, speeds, lap_time)
        self.best_result = racing_line
        
        print(f"Optimal lap time: {lap_time:.2f}s")
        
        return racing_line
    
    def optimize_iterative(self, track: Track, num_iterations: int = 5,
                          steps_per_iteration: int = 20) -> RacingLine:
        """
        Iterative optimization with refinement.
        
        Performs multiple optimization runs, using each result as the
        starting point for the next iteration.
        
        Args:
            track: Track to optimize
            num_iterations: Number of refinement iterations
            steps_per_iteration: Optimization steps per iteration
        
        Returns:
            Final optimized racing line
        """
        current_offsets = None
        best_result = None
        
        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            
            result = self.optimize(track, steps_per_iteration, current_offsets)
            
            if best_result is None or result.lap_time < best_result.lap_time:
                best_result = result
                print(f"New best lap time: {result.lap_time:.2f}s")
            
            # Use this result as starting point for next iteration
            current_offsets = result.offsets
            
            # Add small random perturbation to escape local minima
            if iteration < num_iterations - 1:
                noise_scale = 0.1 * np.std(current_offsets)
                current_offsets += np.random.normal(0, noise_scale, len(current_offsets))
        
        return best_result


if __name__ == "__main__":
    # Demo: Find optimal racing line for our oval
    from ..core.track import create_simple_oval
    
    print("=== Racing Line Optimization Demo ===")
    
    # Create and process track
    oval = create_simple_oval()
    interpolate_track_centerline(oval, target_spacing=3.0)  # Coarser for faster optimization
    
    # Create F1 car
    f1_car = VehicleModel(mu_friction=1.8, max_accel=12.0, max_brake=15.0)
    
    # Baseline: compute centerline lap time
    centerline_speeds, centerline_time = compute_speed_profile(oval, f1_car)
    print(f"Centerline lap time: {centerline_time:.2f}s")
    
    # Optimize racing line
    optimizer = RacingLineOptimizer(f1_car, smoothness_weight=0.5, curvature_weight=0.1)
    
    racing_line = optimizer.optimize(oval, max_iterations=50)
    
    # Compare results
    improvement = centerline_time - racing_line.lap_time
    improvement_percent = (improvement / centerline_time) * 100
    
    print(f"\n=== Optimization Results ===")
    print(f"Centerline time: {centerline_time:.2f}s")
    print(f"Optimal line time: {racing_line.lap_time:.2f}s")
    print(f"Improvement: {improvement:.2f}s ({improvement_percent:.1f}%)")
    
    # Performance summary
    summary = racing_line.get_performance_summary()
    print(f"\n=== Racing Line Analysis ===")
    print(f"Average speed: {summary['avg_speed_kmh']:.1f} km/h")
    print(f"Speed range: {summary['min_speed_kmh']:.1f} - {summary['max_speed_kmh']:.1f} km/h")
    print(f"Max offset: {summary['max_offset']:.1f}m")
    print(f"Line smoothness: {summary['line_smoothness']:.3f}")
