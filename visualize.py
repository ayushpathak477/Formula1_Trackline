"""
Visualization system for racing lines and track analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import sys
sys.path.append('.')

from src.core.track import Track
from src.core.geometry import interpolate_track_centerline, compute_track_boundaries
from src.core.physics import VehicleModel, compute_speed_profile

def plot_track_and_racing_line(track, racing_line_offsets=None, speeds=None, 
                              title="Racing Track Analysis", save_path=None):
    """
    Create a comprehensive visualization of track and racing line.
    
    Args:
        track: Track object with samples
        racing_line_offsets: Optional lateral offsets for racing line
        speeds: Optional speed profile for color-coding
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract track data
    centerline_x = [s.x for s in track.samples]
    centerline_y = [s.y for s in track.samples]
    
    # Compute track boundaries
    inner_boundary, outer_boundary = compute_track_boundaries(track)
    
    # Plot track boundaries
    ax.fill(outer_boundary[:, 0], outer_boundary[:, 1], 
            color='lightgray', alpha=0.8, label='Track surface')
    ax.fill(inner_boundary[:, 0], inner_boundary[:, 1], 
            color='white', alpha=1.0)
    
    # Plot track boundaries as lines
    ax.plot(outer_boundary[:, 0], outer_boundary[:, 1], 'k-', linewidth=2, label='Track limits')
    ax.plot(inner_boundary[:, 0], inner_boundary[:, 1], 'k-', linewidth=2)
    
    # Plot centerline
    ax.plot(centerline_x, centerline_y, 'k--', linewidth=1, alpha=0.6, label='Centerline')
    
    # Plot racing line if provided
    if racing_line_offsets is not None:
        racing_line_x = []
        racing_line_y = []
        
        for i, sample in enumerate(track.samples):
            offset = racing_line_offsets[i]
            line_x = sample.x + offset * sample.normal_x
            line_y = sample.y + offset * sample.normal_y
            racing_line_x.append(line_x)
            racing_line_y.append(line_y)
        
        if speeds is not None:
            # Color-code by speed
            speeds_kmh = np.array(speeds) * 3.6
            
            # Create speed colormap
            colors = plt.cm.plasma(np.linspace(0, 1, len(speeds_kmh)))
            
            # Plot racing line segments with speed colors
            for i in range(len(racing_line_x) - 1):
                ax.plot([racing_line_x[i], racing_line_x[i+1]], 
                       [racing_line_y[i], racing_line_y[i+1]], 
                       color=colors[i], linewidth=3, alpha=0.8)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='plasma', 
                                     norm=plt.Normalize(vmin=speeds_kmh.min(), vmax=speeds_kmh.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Speed (km/h)', fontsize=12)
            
            ax.plot([], [], color='purple', linewidth=3, label='Racing line (colored by speed)')
        else:
            # Simple racing line
            ax.plot(racing_line_x, racing_line_y, 'r-', linewidth=3, label='Optimal racing line')
    
    # Mark start/finish line
    if track.samples:
        start_x, start_y = track.samples[0].x, track.samples[0].y
        ax.plot(start_x, start_y, 'go', markersize=10, label='Start/Finish')
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()

def plot_speed_analysis(track, centerline_speeds, racing_line_speeds=None):
    """Plot speed comparison along track length."""
    distances = [s.s for s in track.samples]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Speed vs distance
    ax1.plot(distances, np.array(centerline_speeds) * 3.6, 'k--', 
             linewidth=2, label='Centerline')
    
    if racing_line_speeds is not None:
        ax1.plot(distances, np.array(racing_line_speeds) * 3.6, 'r-', 
                 linewidth=2, label='Racing line')
        
        # Speed difference
        speed_diff = (np.array(racing_line_speeds) - np.array(centerline_speeds)) * 3.6
        ax2.plot(distances, speed_diff, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.fill_between(distances, speed_diff, 0, where=(speed_diff >= 0), 
                        color='green', alpha=0.3, label='Speed gain')
        ax2.fill_between(distances, speed_diff, 0, where=(speed_diff < 0), 
                        color='red', alpha=0.3, label='Speed loss')
    
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title('Speed Profile Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Distance along track (m)')
    ax2.set_ylabel('Speed difference (km/h)')
    ax2.set_title('Racing Line Speed Advantage')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Demo visualization with our working example
    from debug_optimizer import create_simple_hairpin_track, optimize_simple
    
    print("Creating visualization demo...")
    
    # Create track
    track = create_simple_hairpin_track()
    interpolate_track_centerline(track, target_spacing=5.0)
    
    # Vehicle model
    f1_car = VehicleModel(mu_friction=1.8, max_accel=10.0, max_brake=12.0)
    
    # Get centerline performance
    centerline_speeds, centerline_time = compute_speed_profile(track, f1_car)
    
    # Optimize racing line
    racing_offsets, racing_time = optimize_simple(track, f1_car, max_iter=10)
    
    # Get racing line speeds
    # Quick hack: compute racing line speeds by creating modified track
    modified_samples = []
    for i, sample in enumerate(track.samples):
        new_sample = type(sample)(
            s=sample.s,
            x=sample.x + racing_offsets[i] * sample.normal_x,
            y=sample.y + racing_offsets[i] * sample.normal_y,
            width=sample.width,
            tangent_x=sample.tangent_x,
            tangent_y=sample.tangent_y,
            normal_x=sample.normal_x,
            normal_y=sample.normal_y
        )
        modified_samples.append(new_sample)
    
    temp_track = type(track)(track.control_points, track.closed)
    temp_track.samples = modified_samples
    temp_track.total_length = track.total_length
    
    racing_speeds, _ = compute_speed_profile(temp_track, f1_car)
    
    # Create visualizations
    print("\nCreating track visualization...")
    plot_track_and_racing_line(track, racing_offsets, racing_speeds,
                              title=f"Optimal Racing Line (Improvement: {centerline_time-racing_time:.2f}s)")
    
    print("Creating speed analysis...")
    plot_speed_analysis(track, centerline_speeds, racing_speeds)
