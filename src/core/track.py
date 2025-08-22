"""
Track representation for Formula 1 racing line optimization.

A track is represented as:
- A centerline defined by control points
- A width function (can be constant or variable)
- Interpolated boundaries computed from the centerline

Key concepts:
- Control points: User-defined points that define the track shape
- Centerline: Smooth interpolated curve through control points
- Arc length parameterization: Distance along the track (s)
- Track boundaries: Inner and outer limits computed from centerline + width
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json


@dataclass
class ControlPoint:
    """A user-defined point that shapes the track."""
    x: float
    y: float
    width: float = 10.0  # Track width at this point (meters)


@dataclass
class TrackSample:
    """A discrete sample point along the track centerline."""
    s: float      # Arc length position (meters from start)
    x: float      # X coordinate (meters)
    y: float      # Y coordinate (meters)
    width: float  # Track width at this position (meters)
    tangent_x: float  # Unit tangent vector X component
    tangent_y: float  # Unit tangent vector Y component
    normal_x: float   # Unit normal vector X component (points left)
    normal_y: float   # Unit normal vector Y component (points left)


class Track:
    """
    Represents a racing track with centerline and boundaries.
    
    The track is built from control points and interpolated into
    a smooth centerline with computed boundaries.
    """
    
    def __init__(self, control_points: List[ControlPoint], closed: bool = True):
        """
        Initialize track from control points.
        
        Args:
            control_points: List of points defining the track shape
            closed: Whether the track forms a closed loop (default True for F1)
        """
        self.control_points = control_points
        self.closed = closed
        self.samples: List[TrackSample] = []
        self.total_length: float = 0.0
        
        # These will be computed when we generate the centerline
        self._inner_boundary: Optional[np.ndarray] = None
        self._outer_boundary: Optional[np.ndarray] = None
    
    def get_centerline_points(self) -> np.ndarray:
        """Get the raw control points as a numpy array."""
        points = np.array([[cp.x, cp.y] for cp in self.control_points])
        
        # For closed tracks, add the first point at the end for interpolation
        if self.closed:
            points = np.vstack([points, points[0]])
        
        return points
    
    def get_width_at_points(self) -> np.ndarray:
        """Get track width at each control point."""
        widths = np.array([cp.width for cp in self.control_points])
        
        # For closed tracks, repeat the first width
        if self.closed:
            widths = np.append(widths, widths[0])
        
        return widths
    
    def is_valid(self) -> bool:
        """Check if the track definition is valid."""
        if len(self.control_points) < 3:
            return False
        
        # Check for reasonable track widths
        for cp in self.control_points:
            if cp.width <= 0 or cp.width > 100:  # 0-100m seems reasonable
                return False
        
        return True
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of the track: (min_x, max_x, min_y, max_y)."""
        if not self.control_points:
            return (0, 0, 0, 0)
        
        xs = [cp.x for cp in self.control_points]
        ys = [cp.y for cp in self.control_points]
        
        return (min(xs), max(xs), min(ys), max(ys))
    
    def save_to_file(self, filename: str):
        """Save track definition to JSON file."""
        data = {
            'control_points': [
                {'x': cp.x, 'y': cp.y, 'width': cp.width}
                for cp in self.control_points
            ],
            'closed': self.closed
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'Track':
        """Load track definition from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        control_points = [
            ControlPoint(cp['x'], cp['y'], cp.get('width', 10.0))
            for cp in data['control_points']
        ]
        
        return cls(control_points, data.get('closed', True))
    
    def __repr__(self) -> str:
        return f"Track({len(self.control_points)} points, closed={self.closed})"


def create_simple_oval(width: float = 200.0, height: float = 100.0, 
                      track_width: float = 12.0) -> Track:
    """
    Create a simple oval track for testing.
    
    Args:
        width: Oval width in meters
        height: Oval height in meters  
        track_width: Track width in meters
    
    Returns:
        Track object representing the oval
    """
    # Create an oval with 8 control points
    control_points = []
    
    # Top straight
    control_points.append(ControlPoint(-width/2, height/2, track_width))
    control_points.append(ControlPoint(width/2, height/2, track_width))
    
    # Right turn
    control_points.append(ControlPoint(width/2 + height/4, height/4, track_width))
    control_points.append(ControlPoint(width/2 + height/4, -height/4, track_width))
    
    # Bottom straight
    control_points.append(ControlPoint(width/2, -height/2, track_width))
    control_points.append(ControlPoint(-width/2, -height/2, track_width))
    
    # Left turn
    control_points.append(ControlPoint(-width/2 - height/4, -height/4, track_width))
    control_points.append(ControlPoint(-width/2 - height/4, height/4, track_width))
    
    return Track(control_points, closed=True)


if __name__ == "__main__":
    # Demo: Create and save a simple oval
    oval = create_simple_oval()
    print(f"Created {oval}")
    print(f"Track bounds: {oval.get_bounds()}")
    print(f"Valid: {oval.is_valid()}")
    
    # Save to file
    oval.save_to_file("example_oval.json")
    print("Saved oval to example_oval.json")
