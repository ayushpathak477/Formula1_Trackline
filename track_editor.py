"""
Interactive track editor with racing line optimization.
Click to place points, see optimal line in real-time.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
sys.path.append('.')

from src.core.track import Track, ControlPoint
from src.core.geometry import interpolate_track_centerline, compute_track_boundaries
from src.core.physics import VehicleModel, compute_speed_profile
from debug_optimizer import optimize_simple

class TrackEditor:
    """Interactive track editor with live racing line optimization."""
    
    def __init__(self):
        self.control_points = []
        self.track = None
        self.vehicle = VehicleModel(mu_friction=1.8, max_accel=10.0, max_brake=12.0)
        
        # Setup interactive plot
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.ax.set_xlim(-50, 150)
        self.ax.set_ylim(-100, 50)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Track Editor: Click to place points, Space to optimize racing line')
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Plot elements
        self.control_point_plot = None
        self.track_boundary_plot = None
        self.centerline_plot = None
        self.racing_line_plot = None
        
        self.show_instructions()
    
    def show_instructions(self):
        """Show usage instructions."""
        instructions = [
            "🏁 RACING LINE TRACK EDITOR",
            "",
            "Controls:",
            "• Left click: Add control point",
            "• Right click: Remove last point", 
            "• SPACE: Optimize racing line",
            "• 'c': Clear all points",
            "• 's': Save track",
            "",
            "Tips:",
            "• Need at least 4 points for track",
            "• Make tight corners for best optimization",
            "• Click close to start point to close loop"
        ]
        
        for i, text in enumerate(instructions):
            self.ax.text(0.02, 0.98 - i*0.04, text, transform=self.ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    def on_click(self, event):
        """Handle mouse clicks to add/remove points."""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left click - add point
            self.add_control_point(event.xdata, event.ydata)
        elif event.button == 3:  # Right click - remove last point
            self.remove_last_point()
    
    def on_key(self, event):
        """Handle keyboard input."""
        if event.key == ' ':  # Space - optimize racing line
            self.optimize_racing_line()
        elif event.key == 'c':  # Clear all points
            self.clear_track()
        elif event.key == 's':  # Save track
            self.save_track()
    
    def add_control_point(self, x, y):
        """Add a new control point."""
        # Check if clicking near start point to close loop
        if len(self.control_points) >= 4:
            start_x, start_y = self.control_points[0].x, self.control_points[0].y
            distance = np.sqrt((x - start_x)**2 + (y - start_y)**2)
            if distance < 10:  # Close enough to start
                print(f"Closed track with {len(self.control_points)} points")
                self.generate_track()
                return
        
        # Add new point
        self.control_points.append(ControlPoint(x, y, width=12.0))
        print(f"Added point {len(self.control_points)}: ({x:.1f}, {y:.1f})")
        
        self.update_display()
        
        # Auto-generate track if we have enough points
        if len(self.control_points) >= 4:
            self.generate_track()
    
    def remove_last_point(self):
        """Remove the last control point."""
        if self.control_points:
            removed = self.control_points.pop()
            print(f"Removed point: ({removed.x:.1f}, {removed.y:.1f})")
            self.update_display()
            
            if len(self.control_points) >= 4:
                self.generate_track()
            else:
                self.clear_track_display()
    
    def clear_track(self):
        """Clear all points and track."""
        self.control_points = []
        self.track = None
        self.clear_track_display()
        print("Cleared all points")
    
    def generate_track(self):
        """Generate track geometry from control points."""
        if len(self.control_points) < 4:
            return
        
        # Validate geometry first
        is_valid, message = self.validate_track_geometry()
        if not is_valid:
            print(f"⚠️  Track validation failed: {message}")
            self.ax.set_title(f'Track Editor: {message}')
            self.fig.canvas.draw()
            return
        
        try:
            self.track = Track(self.control_points.copy(), closed=True)
            interpolate_track_centerline(self.track, target_spacing=4.0)
            print(f"✅ Generated valid track: {len(self.track.samples)} samples, {self.track.total_length:.1f}m")
            self.ax.set_title('Track Editor: Valid track! Press SPACE to optimize racing line')
            self.update_display()
        except Exception as e:
            print(f"❌ Error generating track: {e}")
            self.ax.set_title(f'Track Editor: Generation error - {e}')
            self.fig.canvas.draw()
    
    def validate_track_geometry(self):
        """Check if track has overlapping sections or invalid geometry."""
        if len(self.control_points) < 4:
            return True, "Need at least 4 points"
        
        # Check for self-intersections in control point polygon
        points = [(cp.x, cp.y) for cp in self.control_points]
        
        # Simple intersection check for adjacent segments
        for i in range(len(points)):
            for j in range(i + 2, len(points)):
                if j == len(points) - 1 and i == 0:
                    continue  # Skip last-to-first connection
                
                # Check if segments intersect
                p1, p2 = points[i], points[(i + 1) % len(points)]
                p3, p4 = points[j], points[(j + 1) % len(points)]
                
                if self.segments_intersect(p1, p2, p3, p4):
                    return False, f"Track overlaps at points {i+1}-{i+2} and {j+1}-{j+2}"
        
        # Check minimum track dimensions
        xs = [cp.x for cp in self.control_points]
        ys = [cp.y for cp in self.control_points]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        if width < 20 or height < 20:
            return False, "Track too small (minimum 20m x 20m)"
        
        return True, "Track geometry valid"
    
    def segments_intersect(self, p1, p2, p3, p4):
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def optimize_racing_line(self):
        """Compute and display optimal racing line."""
        if not self.track or len(self.track.samples) < 10:
            print("Need a valid track to optimize!")
            return
        
        print("Optimizing racing line...")
        try:
            # Get baseline time
            centerline_speeds, centerline_time = compute_speed_profile(self.track, self.vehicle)
            
            # Optimize
            racing_offsets, racing_time = optimize_simple(self.track, self.vehicle, max_iter=15)
            
            # Show results
            improvement = centerline_time - racing_time
            improvement_percent = (improvement / centerline_time) * 100
            
            print(f"Optimization complete!")
            print(f"Centerline: {centerline_time:.2f}s")
            print(f"Racing line: {racing_time:.2f}s") 
            print(f"Improvement: {improvement:.2f}s ({improvement_percent:.1f}%)")
            
            # Update display with racing line
            self.display_racing_line(racing_offsets)
            
        except Exception as e:
            print(f"Optimization failed: {e}")
    
    def update_display(self):
        """Update the visual display."""
        # Clear previous plots
        self.clear_track_display()
        
        # Plot control points
        if self.control_points:
            xs = [cp.x for cp in self.control_points]
            ys = [cp.y for cp in self.control_points]
            
            self.control_point_plot = self.ax.plot(xs, ys, 'ro-', markersize=8, 
                                                  linewidth=2, alpha=0.7, label='Control points')[0]
        
        # Plot track if available
        if self.track and self.track.samples:
            # Track boundaries
            inner_boundary, outer_boundary = compute_track_boundaries(self.track)
            
            self.track_boundary_plot = [
                self.ax.fill(outer_boundary[:, 0], outer_boundary[:, 1], 
                           color='lightgray', alpha=0.6, label='Track surface'),
                self.ax.fill(inner_boundary[:, 0], inner_boundary[:, 1], 
                           color='white', alpha=1.0),
                self.ax.plot(outer_boundary[:, 0], outer_boundary[:, 1], 'k-', linewidth=2),
                self.ax.plot(inner_boundary[:, 0], inner_boundary[:, 1], 'k-', linewidth=2)
            ]
            
            # Centerline
            centerline_x = [s.x for s in self.track.samples]
            centerline_y = [s.y for s in self.track.samples]
            self.centerline_plot = self.ax.plot(centerline_x, centerline_y, 'k--', 
                                              linewidth=1, alpha=0.6, label='Centerline')[0]
        
        self.ax.legend()
        self.fig.canvas.draw()
    
    def display_racing_line(self, racing_offsets):
        """Display the optimized racing line with speed colors."""
        if not self.track:
            return
        
        # Compute racing line coordinates and speeds
        racing_line_x = []
        racing_line_y = []
        
        for i, sample in enumerate(self.track.samples):
            offset = racing_offsets[i]
            line_x = sample.x + offset * sample.normal_x
            line_y = sample.y + offset * sample.normal_y
            racing_line_x.append(line_x)
            racing_line_y.append(line_y)
        
        # Get speeds for color coding
        try:
            # Create temporary track for speed calculation
            modified_samples = []
            for i, sample in enumerate(self.track.samples):
                new_sample = type(sample)(
                    s=sample.s, x=sample.x + racing_offsets[i] * sample.normal_x,
                    y=sample.y + racing_offsets[i] * sample.normal_y,
                    width=sample.width, tangent_x=sample.tangent_x, tangent_y=sample.tangent_y,
                    normal_x=sample.normal_x, normal_y=sample.normal_y
                )
                modified_samples.append(new_sample)
            
            temp_track = type(self.track)(self.track.control_points, self.track.closed)
            temp_track.samples = modified_samples
            temp_track.total_length = self.track.total_length
            
            speeds, _ = compute_speed_profile(temp_track, self.vehicle)
            speeds_kmh = np.array(speeds) * 3.6
            
            # Plot with speed colors
            colors = plt.cm.plasma(np.linspace(0, 1, len(speeds_kmh)))
            
            for i in range(len(racing_line_x) - 1):
                self.ax.plot([racing_line_x[i], racing_line_x[i+1]], 
                           [racing_line_y[i], racing_line_y[i+1]], 
                           color=colors[i], linewidth=4, alpha=0.9)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='plasma', 
                                     norm=plt.Normalize(vmin=speeds_kmh.min(), vmax=speeds_kmh.max()))
            sm.set_array([])
            if hasattr(self, 'colorbar'):
                self.colorbar.remove()
            self.colorbar = plt.colorbar(sm, ax=self.ax, shrink=0.6)
            self.colorbar.set_label('Speed (km/h)', fontsize=10)
            
        except Exception as e:
            # Fallback: simple red line
            self.ax.plot(racing_line_x, racing_line_y, 'r-', linewidth=4, 
                        alpha=0.9, label='Racing line')
            print(f"Speed visualization failed: {e}")
        
        self.ax.legend()
        self.fig.canvas.draw()
    
    def clear_track_display(self):
        """Clear track-related plot elements."""
        # Clear control points
        if self.control_point_plot:
            self.control_point_plot.remove()
            self.control_point_plot = None
        
        # Clear track boundaries (handle list of plot elements)
        if self.track_boundary_plot:
            for plot_element in self.track_boundary_plot:
                if hasattr(plot_element, 'remove'):
                    plot_element.remove()
                elif isinstance(plot_element, list):
                    for sub_element in plot_element:
                        if hasattr(sub_element, 'remove'):
                            sub_element.remove()
            self.track_boundary_plot = None
        
        # Clear centerline
        if self.centerline_plot:
            self.centerline_plot.remove()
            self.centerline_plot = None
        
        # Clear racing line
        if self.racing_line_plot:
            self.racing_line_plot.remove()
            self.racing_line_plot = None
        
        # Clear any existing racing line segments
        for line in self.ax.lines[:]:
            if line.get_linewidth() == 4:  # Racing line thickness
                line.remove()
    
    def save_track(self):
        """Save current track to JSON file."""
        if not self.track:
            print("No track to save!")
            return
        
        filename = f"custom_track_{len(self.control_points)}points.json"
        self.track.save_to_file(filename)
        print(f"Saved track to {filename}")
    
    def run(self):
        """Start the interactive editor."""
        plt.show()

if __name__ == "__main__":
    print("Starting Interactive Track Editor...")
    print("Click to place points, Space to optimize racing line!")
    
    editor = TrackEditor()
    editor.run()
