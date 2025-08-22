"""
Streamlit Web App for F1 Racing Line Optimization
Professional UI for track design and racing line analysis
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.core.track import Track, ControlPoint
from src.core.geometry import interpolate_track_centerline, compute_track_boundaries
from src.core.physics import VehicleModel, compute_speed_profile
from debug_optimizer import optimize_simple

# Page config
st.set_page_config(
    page_title="F1 Racing Line Optimizer", 
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e10600;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #e10600, #ff4d6d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: #e10600;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'control_points' not in st.session_state:
        st.session_state.control_points = []
    if 'track' not in st.session_state:
        st.session_state.track = None
    if 'racing_line_result' not in st.session_state:
        st.session_state.racing_line_result = None

def create_interactive_plot():
    """Create interactive plotly figure for track editing"""
    fig = go.Figure()
    
    # Plot control points if any
    if st.session_state.control_points:
        points = st.session_state.control_points
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        
        # Control points
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers+lines',
            marker=dict(size=12, color='red', symbol='circle'),
            line=dict(color='red', width=2, dash='dash'),
            name='Control Points',
            hovertemplate='Point %{pointNumber+1}<br>X: %{x:.1f}m<br>Y: %{y:.1f}m<extra></extra>'
        ))
    
    # Plot track if generated
    if st.session_state.track and st.session_state.track.samples:
        track = st.session_state.track
        
        # Track boundaries
        inner_boundary, outer_boundary = compute_track_boundaries(track)
        
        # Outer boundary
        fig.add_trace(go.Scatter(
            x=outer_boundary[:, 0], y=outer_boundary[:, 1],
            mode='lines',
            line=dict(color='black', width=3),
            name='Track Boundary',
            fill=None
        ))
        
        # Inner boundary
        fig.add_trace(go.Scatter(
            x=inner_boundary[:, 0], y=inner_boundary[:, 1],
            mode='lines',
            line=dict(color='black', width=3),
            name='Track Boundary',
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.5)'
        ))
        
        # Centerline
        centerline_x = [s.x for s in track.samples]
        centerline_y = [s.y for s in track.samples]
        fig.add_trace(go.Scatter(
            x=centerline_x, y=centerline_y,
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            name='Centerline'
        ))
        
        # Racing line if optimized
        if st.session_state.racing_line_result:
            offsets, speeds, lap_time = st.session_state.racing_line_result
            
            # Compute racing line coordinates
            racing_x, racing_y = [], []
            for i, sample in enumerate(track.samples):
                offset = offsets[i]
                x = sample.x + offset * sample.normal_x
                y = sample.y + offset * sample.normal_y
                racing_x.append(x)
                racing_y.append(y)
            
            # Speed colormap
            speeds_kmh = np.array(speeds) * 3.6
            
            fig.add_trace(go.Scatter(
                x=racing_x, y=racing_y,
                mode='lines+markers',
                line=dict(width=6),
                marker=dict(
                    size=8,
                    color=speeds_kmh,
                    colorscale='Plasma',
                    colorbar=dict(title="Speed (km/h)", x=1.02),
                    showscale=True
                ),
                name='Racing Line',
                hovertemplate='Speed: %{marker.color:.1f} km/h<br>X: %{x:.1f}m<br>Y: %{y:.1f}m<extra></extra>'
            ))
    
    # Layout
    fig.update_layout(
        title="Interactive Track Editor",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        height=600,
        showlegend=True,
        hovermode='closest',
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal aspect ratio
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def validate_track():
    """Validate current track configuration"""
    if len(st.session_state.control_points) < 4:
        return False, "Need at least 4 control points"
    
    # Check for reasonable spacing
    points = [(p.x, p.y) for p in st.session_state.control_points]
    total_perimeter = 0
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        total_perimeter += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    if total_perimeter < 50:
        return False, "Track too small (minimum 50m perimeter)"
    
    return True, "Track valid"

def main():
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🏎️ F1 Racing Line Optimizer</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("Track Designer")
    
    # Manual point entry
    st.sidebar.subheader("Add Control Points")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x_coord = st.number_input("X (m)", value=0.0, step=5.0, key="x_input")
    with col2:
        y_coord = st.number_input("Y (m)", value=0.0, step=5.0, key="y_input")
    
    track_width = st.sidebar.slider("Track Width (m)", 8.0, 20.0, 12.0, 0.5)
    
    if st.sidebar.button("Add Point", type="primary"):
        new_point = ControlPoint(x_coord, y_coord, track_width)
        st.session_state.control_points.append(new_point)
        st.rerun()
    
    # Quick track templates
    st.sidebar.subheader("Quick Templates")
    if st.sidebar.button("Simple Oval"):
        st.session_state.control_points = [
            ControlPoint(-50, 25, track_width),
            ControlPoint(50, 25, track_width),
            ControlPoint(70, 0, track_width),
            ControlPoint(50, -25, track_width),
            ControlPoint(-50, -25, track_width),
            ControlPoint(-70, 0, track_width)
        ]
        st.rerun()
    
    if st.sidebar.button("Hairpin Track"):
        st.session_state.control_points = [
            ControlPoint(0, 0, track_width),
            ControlPoint(50, 0, track_width),
            ControlPoint(65, -15, track_width),
            ControlPoint(65, -45, track_width),
            ControlPoint(50, -60, track_width),
            ControlPoint(0, -65, track_width),
            ControlPoint(-15, -50, track_width),
            ControlPoint(-15, -15, track_width)
        ]
        st.rerun()
    
    # Track management
    st.sidebar.subheader("Track Management")
    if st.sidebar.button("Clear Track", type="secondary"):
        st.session_state.control_points = []
        st.session_state.track = None
        st.session_state.racing_line_result = None
        st.rerun()
    
    if st.sidebar.button("Remove Last Point"):
        if st.session_state.control_points:
            st.session_state.control_points.pop()
            st.rerun()
    
    # Vehicle parameters
    st.sidebar.subheader("Vehicle Setup")
    mu = st.sidebar.slider("Friction Coefficient", 1.0, 2.5, 1.8, 0.1)
    max_accel = st.sidebar.slider("Max Acceleration (m/s²)", 8.0, 15.0, 12.0, 0.5)
    max_brake = st.sidebar.slider("Max Braking (m/s²)", 10.0, 20.0, 15.0, 0.5)
    max_speed = st.sidebar.slider("Top Speed (m/s)", 60.0, 120.0, 100.0, 5.0)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Track visualization
        fig = create_interactive_plot()
        st.plotly_chart(fig, use_container_width=True)
        
        # Track info
        if st.session_state.control_points:
            st.info(f"📍 Control Points: {len(st.session_state.control_points)}")
            
            is_valid, message = validate_track()
            if is_valid:
                st.success(f"✅ {message}")
                
                # Generate track
                if st.button("Generate Track", type="primary"):
                    try:
                        track = Track(st.session_state.control_points.copy(), closed=True)
                        interpolate_track_centerline(track, target_spacing=4.0)
                        st.session_state.track = track
                        st.success(f"🏁 Track generated: {len(track.samples)} samples, {track.total_length:.1f}m")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to generate track: {e}")
            else:
                st.warning(f"⚠️ {message}")
    
    with col2:
        # Performance panel
        st.subheader("Performance Analysis")
        
        if st.session_state.track:
            # Vehicle model
            vehicle = VehicleModel(
                mu_friction=mu,
                max_accel=max_accel,
                max_brake=max_brake,
                max_speed=max_speed
            )
            
            # Optimize button
            if st.button("🚀 Optimize Racing Line", type="primary"):
                with st.spinner("Optimizing racing line..."):
                    try:
                        # Baseline
                        centerline_speeds, centerline_time = compute_speed_profile(st.session_state.track, vehicle)
                        
                        # Optimize
                        racing_offsets, racing_time = optimize_simple(st.session_state.track, vehicle, max_iter=15)
                        
                        # Compute racing line speeds
                        modified_samples = []
                        for i, sample in enumerate(st.session_state.track.samples):
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
                        
                        temp_track = Track(st.session_state.track.control_points, st.session_state.track.closed)
                        temp_track.samples = modified_samples
                        temp_track.total_length = st.session_state.track.total_length
                        
                        racing_speeds, _ = compute_speed_profile(temp_track, vehicle)
                        
                        # Store results
                        st.session_state.racing_line_result = (racing_offsets, racing_speeds, racing_time)
                        
                        # Show results
                        improvement = centerline_time - racing_time
                        improvement_percent = (improvement / centerline_time) * 100
                        
                        st.success("Optimization Complete!")
                        
                        # Metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Centerline Time", f"{centerline_time:.2f}s")
                            st.metric("Racing Line Time", f"{racing_time:.2f}s")
                        with col2:
                            st.metric("Improvement", f"{improvement:.2f}s", f"{improvement_percent:.1f}%")
                            st.metric("Max Offset", f"{np.max(np.abs(racing_offsets)):.1f}m")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
            
            # Results display
            if st.session_state.racing_line_result:
                offsets, speeds, lap_time = st.session_state.racing_line_result
                
                st.subheader("📊 Results")
                
                # Speed analysis chart
                distances = [s.s for s in st.session_state.track.samples]
                centerline_speeds, _ = compute_speed_profile(st.session_state.track, vehicle)
                
                speed_fig = go.Figure()
                speed_fig.add_trace(go.Scatter(
                    x=distances, 
                    y=np.array(centerline_speeds) * 3.6,
                    mode='lines',
                    name='Centerline',
                    line=dict(color='gray', dash='dash')
                ))
                speed_fig.add_trace(go.Scatter(
                    x=distances, 
                    y=np.array(speeds) * 3.6,
                    mode='lines',
                    name='Racing Line',
                    line=dict(color='red', width=3)
                ))
                
                speed_fig.update_layout(
                    title="Speed Profile Comparison",
                    xaxis_title="Distance (m)",
                    yaxis_title="Speed (km/h)",
                    height=300
                )
                
                st.plotly_chart(speed_fig, use_container_width=True)
        else:
            st.info("👆 Add control points and generate a track to start optimization")

if __name__ == "__main__":
    main()
