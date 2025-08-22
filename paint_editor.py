"""
Paint-like Track Drawing Tool
Smooth, intuitive track creation with mouse drawing
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.core.track import Track, ControlPoint
from src.core.geometry import interpolate_track_centerline, compute_track_boundaries
from src.core.physics import VehicleModel, compute_speed_profile
from debug_optimizer import optimize_simple

class TrackDrawingApp:
    """Paint-like track drawing application with racing line optimization"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🏎️ F1 Track Designer - Paint Mode")
        self.root.geometry("1400x900")
        
        # Drawing state
        self.drawing = False
        self.drawn_points = []
        self.track = None
        self.racing_line_result = None
        
        # Vehicle model
        self.vehicle = VehicleModel(mu_friction=1.8, max_accel=12.0, max_brake=15.0)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="🎨 Track Designer", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Instructions
        instructions = tk.Text(control_frame, height=8, width=30, wrap=tk.WORD)
        instructions.insert(tk.END, """🏁 F1 Track Designer

HOW TO USE:
1. Click and drag to draw track
2. Close loop by drawing near start
3. Click 'Generate Track' 
4. Click 'Optimize Racing Line'

CONTROLS:
• Left Mouse: Draw track
• Right Mouse: Erase mode
• Clear: Start over
• Save/Load: Track files""")
        instructions.config(state=tk.DISABLED)
        instructions.pack(pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.draw_btn = ttk.Button(button_frame, text="🎨 Draw Mode", command=self.toggle_draw_mode)
        self.draw_btn.pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="🗑️ Clear Track", command=self.clear_track).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🏁 Generate Track", command=self.generate_track).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🚀 Optimize Line", command=self.optimize_racing_line).pack(fill=tk.X, pady=2)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Vehicle parameters
        ttk.Label(control_frame, text="🏎️ Vehicle Setup", font=('Arial', 12, 'bold')).pack()
        
        # Friction coefficient
        ttk.Label(control_frame, text="Friction Coefficient").pack(anchor=tk.W)
        self.friction_var = tk.DoubleVar(value=1.8)
        friction_scale = ttk.Scale(control_frame, from_=1.0, to=2.5, variable=self.friction_var, orient=tk.HORIZONTAL)
        friction_scale.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(control_frame, textvariable=self.friction_var).pack()
        
        # Max acceleration
        ttk.Label(control_frame, text="Max Acceleration (m/s²)").pack(anchor=tk.W)
        self.accel_var = tk.DoubleVar(value=12.0)
        accel_scale = ttk.Scale(control_frame, from_=8.0, to=15.0, variable=self.accel_var, orient=tk.HORIZONTAL)
        accel_scale.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(control_frame, textvariable=self.accel_var).pack()
        
        # Max braking
        ttk.Label(control_frame, text="Max Braking (m/s²)").pack(anchor=tk.W)
        self.brake_var = tk.DoubleVar(value=15.0)
        brake_scale = ttk.Scale(control_frame, from_=10.0, to=20.0, variable=self.brake_var, orient=tk.HORIZONTAL)
        brake_scale.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(control_frame, textvariable=self.brake_var).pack()
        
        # Track width
        ttk.Label(control_frame, text="Track Width (m)").pack(anchor=tk.W)
        self.width_var = tk.DoubleVar(value=12.0)
        width_scale = ttk.Scale(control_frame, from_=8.0, to=20.0, variable=self.width_var, orient=tk.HORIZONTAL)
        width_scale.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(control_frame, textvariable=self.width_var).pack()
        
        # Results panel
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="📊 Results", font=('Arial', 12, 'bold')).pack()
        
        self.results_text = tk.Text(control_frame, height=8, width=30)
        self.results_text.pack(pady=5)
        
        # Right panel - Drawing canvas
        canvas_frame = ttk.LabelFrame(main_frame, text="🏁 Track Canvas", padding=5)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Draw Your Track Here")
        
        # Canvas widget
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to draw! Click and drag to create your track.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def toggle_draw_mode(self):
        """Toggle between draw and erase mode"""
        # For now, just drawing mode
        self.status_var.set("Draw mode active - click and drag to draw track")
        
    def on_mouse_press(self, event):
        """Start drawing when mouse pressed"""
        if event.inaxes == self.ax and event.button == 1:  # Left click
            self.drawing = True
            self.drawn_points = [(event.xdata, event.ydata)]
            self.status_var.set("Drawing...")
    
    def on_mouse_move(self, event):
        """Continue drawing while mouse moves"""
        if self.drawing and event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                self.drawn_points.append((event.xdata, event.ydata))
                
                # Real-time drawing feedback
                if len(self.drawn_points) > 1:
                    # Clear previous drawing line
                    for line in self.ax.lines[:]:
                        if line.get_color() == 'blue' and line.get_linewidth() == 3:
                            line.remove()
                    
                    # Draw current line
                    xs, ys = zip(*self.drawn_points)
                    self.ax.plot(xs, ys, 'b-', linewidth=3, alpha=0.7)
                    self.canvas.draw_idle()
    
    def on_mouse_release(self, event):
        """Finish drawing when mouse released"""
        if self.drawing:
            self.drawing = False
            
            if len(self.drawn_points) > 10:  # Minimum points for a track
                # Check if track is closed (near start point)
                start = self.drawn_points[0]
                end = self.drawn_points[-1]
                distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                
                if distance < 20:  # Close enough to start
                    self.drawn_points.append(start)  # Close the loop
                    self.status_var.set(f"Track closed! {len(self.drawn_points)} points drawn. Click 'Generate Track'.")
                else:
                    self.status_var.set(f"Track drawn with {len(self.drawn_points)} points. Draw near start to close loop.")
            else:
                self.status_var.set("Track too short! Draw a longer path.")
    
    def clear_track(self):
        """Clear all drawn elements"""
        self.drawn_points = []
        self.track = None
        self.racing_line_result = None
        
        self.ax.clear()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Draw Your Track Here")
        self.canvas.draw()
        
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Canvas cleared. Ready to draw!")
    
    def generate_track(self):
        """Convert drawn points to track geometry"""
        if len(self.drawn_points) < 10:
            messagebox.showerror("Error", "Draw a track first!")
            return
        
        try:
            # Simplify drawn points to control points
            control_points = self.simplify_drawn_points()
            
            # Create track with boundary validation
            self.track = Track(control_points, closed=True)
            
            # Try different interpolation strategies if needed
            success = False
            for spacing in [4.0, 2.0, 1.0]:  # Try different sample spacings
                try:
                    interpolate_track_centerline(self.track, target_spacing=spacing)
                    
                    # Validate that centerline stays within reasonable bounds
                    if self.validate_centerline():
                        success = True
                        break
                    else:
                        # Clear samples for next attempt
                        self.track.samples = []
                        
                except Exception as e:
                    print(f"Interpolation failed with spacing {spacing}: {e}")
                    continue
            
            if not success:
                # Fall back to simpler interpolation using original drawn points more directly
                self.track = self.create_conservative_track()
            
            # Update vehicle parameters
            self.vehicle = VehicleModel(
                mu_friction=self.friction_var.get(),
                max_accel=self.accel_var.get(),
                max_brake=self.brake_var.get(),
                max_speed=100.0
            )
            
            # Visualize track
            self.visualize_track()
            
            self.status_var.set(f"Track generated! {len(self.track.samples)} samples, {self.track.total_length:.1f}m")
            
            # Update results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"✅ Track Generated\n")
            self.results_text.insert(tk.END, f"Control Points: {len(control_points)}\n")
            self.results_text.insert(tk.END, f"Track Length: {self.track.total_length:.1f}m\n")
            self.results_text.insert(tk.END, f"Samples: {len(self.track.samples)}\n\n")
            self.results_text.insert(tk.END, f"Ready for optimization!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate track: {e}")
    
    def validate_centerline(self):
        """Check if centerline stays within reasonable bounds of the original drawing"""
        if not self.track.samples or not self.drawn_points:
            return True
        
        # Get centerline points
        centerline_points = [(s.x, s.y) for s in self.track.samples]
        
        # Check if any centerline point is too far from the nearest drawn point
        max_allowed_distance = self.width_var.get() * 0.3  # 30% of track width
        
        for cx, cy in centerline_points:
            min_distance = float('inf')
            for dx, dy in self.drawn_points:
                distance = np.sqrt((cx - dx)**2 + (cy - dy)**2)
                min_distance = min(min_distance, distance)
            
            if min_distance > max_allowed_distance:
                return False
        
        return True
    
    def create_conservative_track(self):
        """Create track using a more conservative approach that follows the drawn path closely"""
        from src.core.track import TrackSample
        
        # Use drawn points directly with minimal smoothing
        points = np.array(self.drawn_points[:-1])  # Remove duplicate closing point
        
        # Simple smoothing - just average nearby points
        smoothed = []
        window = 3
        for i in range(len(points)):
            start = max(0, i - window)
            end = min(len(points), i + window + 1)
            avg_point = np.mean(points[start:end], axis=0)
            smoothed.append(avg_point)
        
        # Create track samples directly
        track_width = self.width_var.get()
        samples = []
        total_length = 0.0
        
        for i, point in enumerate(smoothed):
            # Compute tangent direction
            if i == 0:
                tangent = smoothed[1] - smoothed[-1]  # closed loop
            elif i == len(smoothed) - 1:
                tangent = smoothed[0] - smoothed[i-1]  # closed loop
            else:
                tangent = smoothed[i+1] - smoothed[i-1]
            
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 0:
                tangent_unit = tangent / tangent_norm
                normal_unit = np.array([-tangent_unit[1], tangent_unit[0]])
            else:
                tangent_unit = np.array([1.0, 0.0])
                normal_unit = np.array([0.0, 1.0])
            
            sample = TrackSample(
                s=total_length,
                x=point[0],
                y=point[1],
                width=track_width,
                tangent_x=tangent_unit[0],
                tangent_y=tangent_unit[1],
                normal_x=normal_unit[0],
                normal_y=normal_unit[1]
            )
            
            samples.append(sample)
            
            # Update arc length
            if i > 0:
                total_length += np.linalg.norm(point - smoothed[i-1])
        
        # Create new track object
        conservative_track = Track([], closed=True)
        conservative_track.samples = samples
        conservative_track.total_length = total_length
        
        return conservative_track
    
    def simplify_drawn_points(self):
        """Convert dense drawn points to sparse control points"""
        # Douglas-Peucker-like simplification
        points = np.array(self.drawn_points[:-1])  # Remove duplicate closing point
        
        if len(points) < 4:
            raise ValueError("Need at least 4 points to create a track")
        
        # Adaptive simplification based on curvature
        simplified = self.adaptive_simplify(points, max_points=20)
        
        # Ensure we have enough points for smooth interpolation
        if len(simplified) < 6:
            # Fall back to uniform decimation if adaptive method produces too few points
            step = max(1, len(points) // 12)
            simplified = points[::step]
        
        # Convert to ControlPoint objects
        track_width = self.width_var.get()
        control_points = [ControlPoint(p[0], p[1], track_width) for p in simplified]
        
        return control_points
    
    def adaptive_simplify(self, points, max_points=20, tolerance=8.0):
        """Simplify points while preserving important corners"""
        if len(points) <= max_points:
            return points
        
        # Start with the first point
        simplified = [points[0]]
        
        i = 1
        while i < len(points) and len(simplified) < max_points - 1:
            # Look ahead to find a good next control point
            best_idx = i
            max_distance = 0
            
            # Check up to next 15 points or until we find a significant turn
            for j in range(i + 1, min(len(points), i + 15)):
                # Calculate distance from line between last control point and current candidate
                if len(simplified) > 0:
                    line_start = simplified[-1]
                    line_end = points[j]
                    
                    # Find maximum distance of intermediate points from this line
                    max_deviation = 0
                    for k in range(i, j):
                        deviation = self.point_to_line_distance(points[k], line_start, line_end)
                        max_deviation = max(max_deviation, deviation)
                    
                    # If deviation is significant or we've gone far enough, stop here
                    if max_deviation > tolerance or j - i > 8:
                        best_idx = j
                        break
                    
                    best_idx = j
            
            simplified.append(points[best_idx])
            i = best_idx + 1
        
        # Always include the last point
        if simplified[-1] is not points[-1]:
            simplified.append(points[-1])
        
        return np.array(simplified)
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)
        
        line_unit = line_vec / line_len
        projection = np.dot(point_vec, line_unit)
        
        if projection < 0:
            return np.linalg.norm(point_vec)
        elif projection > line_len:
            return np.linalg.norm(point - line_end)
        else:
            closest_point = line_start + projection * line_unit
            return np.linalg.norm(point - closest_point)
    
    def visualize_track(self):
        """Draw the generated track"""
        if not self.track:
            return
        
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Track boundaries
        inner_boundary, outer_boundary = compute_track_boundaries(self.track)
        
        # Fill track surface
        self.ax.fill(outer_boundary[:, 0], outer_boundary[:, 1], 
                    color='lightgray', alpha=0.6, label='Track surface')
        self.ax.fill(inner_boundary[:, 0], inner_boundary[:, 1], 
                    color='white', alpha=1.0)
        
        # Track boundaries
        self.ax.plot(outer_boundary[:, 0], outer_boundary[:, 1], 'k-', linewidth=2)
        self.ax.plot(inner_boundary[:, 0], inner_boundary[:, 1], 'k-', linewidth=2)
        
        # Centerline
        centerline_x = [s.x for s in self.track.samples]
        centerline_y = [s.y for s in self.track.samples]
        self.ax.plot(centerline_x, centerline_y, 'k--', linewidth=1, alpha=0.6, label='Centerline')
        
        # Original drawn path
        if self.drawn_points:
            xs, ys = zip(*self.drawn_points)
            self.ax.plot(xs, ys, 'b:', linewidth=1, alpha=0.5, label='Original drawing')
        
        self.ax.set_title("Generated Track")
        self.ax.legend()
        self.canvas.draw()
    
    def optimize_racing_line(self):
        """Optimize and display racing line"""
        if not self.track:
            messagebox.showerror("Error", "Generate a track first!")
            return
        
        self.status_var.set("Optimizing racing line...")
        self.root.update()
        
        try:
            # Get baseline
            centerline_speeds, centerline_time = compute_speed_profile(self.track, self.vehicle)
            
            # Optimize
            racing_offsets, racing_time = optimize_simple(self.track, self.vehicle, max_iter=15)
            
            # Compute racing line speeds
            modified_samples = []
            for i, sample in enumerate(self.track.samples):
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
            
            temp_track = Track(self.track.control_points, self.track.closed)
            temp_track.samples = modified_samples
            temp_track.total_length = self.track.total_length
            
            racing_speeds, _ = compute_speed_profile(temp_track, self.vehicle)
            
            # Store results
            self.racing_line_result = (racing_offsets, racing_speeds, racing_time)
            
            # Visualize racing line
            self.visualize_racing_line()
            
            # Update results
            improvement = centerline_time - racing_time
            improvement_percent = (improvement / centerline_time) * 100
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"🏁 OPTIMIZATION COMPLETE!\n\n")
            self.results_text.insert(tk.END, f"Centerline Time: {centerline_time:.2f}s\n")
            self.results_text.insert(tk.END, f"Racing Line Time: {racing_time:.2f}s\n")
            self.results_text.insert(tk.END, f"Improvement: {improvement:.2f}s\n")
            self.results_text.insert(tk.END, f"Percentage: {improvement_percent:.1f}%\n\n")
            self.results_text.insert(tk.END, f"Max Offset: {np.max(np.abs(racing_offsets)):.1f}m\n")
            self.results_text.insert(tk.END, f"Min Speed: {np.min(racing_speeds)*3.6:.1f} km/h\n")
            self.results_text.insert(tk.END, f"Max Speed: {np.max(racing_speeds)*3.6:.1f} km/h")
            
            self.status_var.set(f"Optimization complete! {improvement_percent:.1f}% improvement ({improvement:.2f}s)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {e}")
            self.status_var.set("Optimization failed")
    
    def visualize_racing_line(self):
        """Add racing line to visualization"""
        if not self.racing_line_result:
            return
        
        racing_offsets, racing_speeds, _ = self.racing_line_result
        
        # Compute racing line coordinates
        racing_x, racing_y = [], []
        for i, sample in enumerate(self.track.samples):
            offset = racing_offsets[i]
            x = sample.x + offset * sample.normal_x
            y = sample.y + offset * sample.normal_y
            racing_x.append(x)
            racing_y.append(y)
        
        # Color by speed
        speeds_kmh = np.array(racing_speeds) * 3.6
        
        # Plot racing line with speed colors
        for i in range(len(racing_x) - 1):
            speed_ratio = (speeds_kmh[i] - speeds_kmh.min()) / (speeds_kmh.max() - speeds_kmh.min())
            color = plt.cm.plasma(speed_ratio)
            
            self.ax.plot([racing_x[i], racing_x[i+1]], [racing_y[i], racing_y[i+1]], 
                        color=color, linewidth=4, alpha=0.9)
        
        # Add colorbar info to title
        self.ax.set_title(f"Racing Line (Speed: {speeds_kmh.min():.0f}-{speeds_kmh.max():.0f} km/h)")
        
        self.canvas.draw()

def main():
    """Launch the track drawing application"""
    root = tk.Tk()
    app = TrackDrawingApp(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()
