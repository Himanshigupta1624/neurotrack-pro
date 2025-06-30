import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import math
import random

# Initialize MediaPipe Pose with optimized settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # Reduced from default 2 for better performance
    enable_segmentation=False,  # Disable segmentation for speed
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

important_body_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
]


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


class DummyVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.left_angle = 90
        self.right_angle = 90
        self.frame_count = 0
        self.time_offset = time.time()
        
    def transform(self, frame):
        self.frame_count += 1
        
        # Create a dummy frame with exercise simulation
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img.fill(30)  # Dark background
        
        # Add title text
        cv2.putText(img, "CAMERA MODE", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, "Simulating arm exercises...", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Simulate realistic arm movement patterns
        t = (time.time() - self.time_offset) * 0.5  # Slow movement
        
        # Left arm simulation (moving up and down)
        left_base_angle = 90 + 60 * math.sin(t)  # Range: 30-150 degrees
        left_noise = random.uniform(-5, 5)  # Add some realistic noise
        self.left_angle = max(0, min(180, left_base_angle + left_noise))
        
        # Right arm simulation (different pattern)
        right_base_angle = 90 + 45 * math.sin(t * 0.8 + 1)  # Range: 45-135 degrees
        right_noise = random.uniform(-5, 5)
        self.right_angle = max(0, min(180, right_base_angle + right_noise))
        
        # Draw stick figure representation
        center_x, center_y = 320, 240
        
        # Body
        cv2.line(img, (center_x, center_y - 50), (center_x, center_y + 100), (255, 255, 255), 3)
        
        # Head
        cv2.circle(img, (center_x, center_y - 80), 25, (255, 255, 255), 2)
        
        # Calculate arm positions based on angles
        arm_length = 80
        
        # Left arm
        left_rad = math.radians(self.left_angle - 90)  # Convert to radians, adjust for vertical
        left_end_x = center_x - 50 + int(arm_length * math.cos(left_rad))
        left_end_y = center_y - 20 + int(arm_length * math.sin(left_rad))
        cv2.line(img, (center_x - 50, center_y - 20), (left_end_x, left_end_y), (0, 255, 0), 4)
        cv2.circle(img, (center_x - 50, center_y - 20), 8, (0, 255, 0), -1)  # Shoulder
        cv2.circle(img, (left_end_x, left_end_y), 6, (0, 255, 0), -1)  # Hand
        
        # Right arm
        right_rad = math.radians(180 - self.right_angle - 90)
        right_end_x = center_x + 50 + int(arm_length * math.cos(right_rad))
        right_end_y = center_y - 20 + int(arm_length * math.sin(right_rad))
        cv2.line(img, (center_x + 50, center_y - 20), (right_end_x, right_end_y), (0, 0, 255), 4)
        cv2.circle(img, (center_x + 50, center_y - 20), 8, (0, 0, 255), -1)  # Shoulder
        cv2.circle(img, (right_end_x, right_end_y), 6, (0, 0, 255), -1)  # Hand
        
        # Display current angles
        cv2.putText(img, f"Left: {int(self.left_angle)}°", (50, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Right: {int(self.right_angle)}°", (50, 430), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(img, "This simulates rehabilitation exercises", (50, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return img


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.left_angle = 0
        self.right_angle = 0
        self.frame_count = 0
        self.process_every_n_frames = 3  # Process every 3rd frame for better performance

    def transform(self, frame):
        self.frame_count += 1
        
        # Convert frame to BGR format
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Resize frame for faster processing
        height, width = img.shape[:2]
        if width > 640:  # Resize if width is greater than 640px
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Process only every Nth frame to reduce computational load
        if self.frame_count % self.process_every_n_frames == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w = img.shape[:2]

                # Only draw essential connections for performance
                essential_connections = [
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                ]

                # Draw essential connections only
                for start_landmark, end_landmark in essential_connections:
                    start = landmarks[start_landmark.value]
                    end = landmarks[end_landmark.value]
                    if start.visibility > 0.5 and end.visibility > 0.5:
                        x1, y1 = int(start.x * w), int(start.y * h)
                        x2, y2 = int(end.x * w), int(end.y * h)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw only important landmarks
                for idx in important_body_indices:
                    landmark = landmarks[idx]
                    if landmark.visibility > 0.5:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
                        cv2.circle(img, (x, y), 8, (0, 180, 0), 2)

                # Calculate angles efficiently
                try:
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    
                    if left_shoulder.visibility > 0.5 and left_elbow.visibility > 0.5:
                        left_shoulder_coords = [left_shoulder.x * w, left_shoulder.y * h]
                        left_elbow_coords = [left_elbow.x * w, left_elbow.y * h]
                        self.left_angle = 180 - calculate_angle(left_elbow_coords, left_shoulder_coords,
                                                              [left_shoulder_coords[0], left_shoulder_coords[1] - 100])

                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    
                    if right_shoulder.visibility > 0.5 and right_elbow.visibility > 0.5:
                        right_shoulder_coords = [right_shoulder.x * w, right_shoulder.y * h]
                        right_elbow_coords = [right_elbow.x * w, right_elbow.y * h]
                        self.right_angle = 180 - calculate_angle(right_elbow_coords, right_shoulder_coords,
                                                               [right_shoulder_coords[0], right_shoulder_coords[1] - 100])
                except Exception as e:
                    # Handle any potential errors silently to maintain performance
                    pass

        return img


class PrivateVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.left_angle = 90
        self.right_angle = 90
        self.frame_count = 0
        self.process_every_n_frames = 3  # Process every 3rd frame for better performance
        self.privacy_notice_added = False

    def transform(self, frame):
        self.frame_count += 1
        
        # Convert frame to BGR format
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Resize frame for faster processing
        height, width = img.shape[:2]
        if width > 640:  # Resize if width is greater than 640px
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Process only every Nth frame to reduce computational load
        if self.frame_count % self.process_every_n_frames == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w = img.shape[:2]

                # Only draw essential connections for performance
                essential_connections = [
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                ]

                # Draw essential connections only
                for start_landmark, end_landmark in essential_connections:
                    start = landmarks[start_landmark.value]
                    end = landmarks[end_landmark.value]
                    if start.visibility > 0.5 and end.visibility > 0.5:
                        x1, y1 = int(start.x * w), int(start.y * h)
                        x2, y2 = int(end.x * w), int(end.y * h)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw only important landmarks
                for idx in important_body_indices:
                    landmark = landmarks[idx]
                    if landmark.visibility > 0.5:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
                        cv2.circle(img, (x, y), 8, (0, 180, 0), 2)

                # Calculate angles efficiently
                try:
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    
                    if left_shoulder.visibility > 0.5 and left_elbow.visibility > 0.5:
                        left_shoulder_coords = [left_shoulder.x * w, left_shoulder.y * h]
                        left_elbow_coords = [left_elbow.x * w, left_elbow.y * h]
                        self.left_angle = 180 - calculate_angle(left_elbow_coords, left_shoulder_coords,
                                                              [left_shoulder_coords[0], left_shoulder_coords[1] - 100])

                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    
                    if right_shoulder.visibility > 0.5 and right_elbow.visibility > 0.5:
                        right_shoulder_coords = [right_shoulder.x * w, right_shoulder.y * h]
                        right_elbow_coords = [right_elbow.x * w, right_elbow.y * h]
                        self.right_angle = 180 - calculate_angle(right_elbow_coords, right_shoulder_coords,
                                                               [right_shoulder_coords[0], right_shoulder_coords[1] - 100])
                except Exception as e:
                    # Handle any potential errors silently to maintain performance
                    pass
        
        # Add privacy notice to the frame
        if not self.privacy_notice_added or self.frame_count % 100 == 0:  # Refresh the notice periodically
            cv2.rectangle(img, (10, 10), (350, 70), (0, 0, 0), -1)
            cv2.rectangle(img, (10, 10), (350, 70), (0, 255, 0), 2)
            cv2.putText(img, "PRIVATE MODE ACTIVE", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "No data being stored or logged", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.privacy_notice_added = True
            
        # Display current angles
        cv2.rectangle(img, (10, height - 80), (210, height - 10), (0, 0, 0), -1)
        cv2.putText(img, f"Left Arm: {int(self.left_angle)}°", (20, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Right Arm: {int(self.right_angle)}°", (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return img


def draw_angle_meter(angle, label):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))  # Smaller size for better performance
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    circle = plt.Circle((0, 0), 1, color=(0.2, 0.2, 0.2), fill=True)
    ax.add_artist(circle)

    if angle > 120:
        arc_color = "#4CAF50"  # Green
        level = "Excellent"
    elif 60 < angle <= 120:
        arc_color = "#FFC107"  # Amber
        level = "Moderate"
    else:
        arc_color = "#F44336"  # Red
        level = "Needs Work"

    # Simplified arc drawing for better performance
    theta = np.linspace(np.pi, np.pi - (np.pi * (angle / 180)), 50)  # Reduced points
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, color=arc_color, linewidth=6)

    ax.text(0, 0, f"{int(angle)}°", ha='center', va='center', fontsize=16, color='white', fontweight='bold')
    ax.text(0, -1.3, label, ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    ax.text(0, 1.2, level, ha='center', va='center', fontsize=12, color=arc_color, fontweight='bold')

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', transparent=True, dpi=80)  # Reduced DPI
    buf.seek(0)
    plt.close(fig)  # Important: close figure to free memory
    return buf


def main():
    # Custom CSS for styling
    st.set_page_config(
        layout="wide",
        page_title="NeuroTrack Pro | Stroke Therapy Monitoring",
        page_icon="🧠"
    )
    st.markdown("""
    <style>
        .main {
            background-color: #f5f9fc;
        }
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4f0fb 100%);
        }
        .header {
            color: #2c3e50;
            padding: 1rem;
            border-radius: 10px;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .timer {
            font-size: 1.2rem;
            color: #3498db;
            font-weight: bold;
        }
        .download-btn {
            background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stButton>button {
            background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .success-box {
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)


    # Header Section
    st.markdown("""
    <div class="header">
        <h1 style="margin:0; color:#2c3e50;">🧠 NeuroTrack Pro</h1>
        <p style="margin:0; color:#7f8c8d;">AI-Powered Stroke Rehabilitation Progress Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    # Camera selection
    st.markdown("""
    <div class="card">
        <h3 style="color:#2c3e50; margin-bottom:1rem;">Camera Settings</h3>
    """, unsafe_allow_html=True)
    
    camera_mode = st.selectbox(
        "Select Camera Mode:",
        ["Real Camera", "Private Camera (No Tracking)", "Camera (for testing/deployment)"],
        help="Choose Real Camera for full tracking, Private Camera for no data logging, or  Camera for testing"
    )
    
    if camera_mode == "Camera (for testing/deployment)":
        st.markdown("""
        <div style="background-color: #e7f3ff; border: 1px solid #b6d7ff; color: #1e40af; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <strong>🤖  camera mode will simulate realistic arm movement patterns for testing purposes.</strong>
        </div>
        """, unsafe_allow_html=True)
    elif camera_mode == "Private Camera (No Tracking)":
        st.markdown("""
        <div style="background-color: #ecfdf5; border: 1px solid #6ee7b7; color: #065f46; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <strong>🔒 Private mode uses your webcam but doesn't store or track motion data. Your video stays on your device.</strong>
            <p style="margin-top:0.5rem;">Angles are displayed live but not recorded unless you enable tracking.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #f0f9ff; border: 1px solid #7dd3fc; color: #0369a1; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <strong>📹 Real camera mode will use your webcam for live pose detection and full session tracking.</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Data collection optimization
    if "data_collection_interval" not in st.session_state:
        st.session_state.data_collection_interval = 5  # Collect data every 5 frames
    if "frame_counter" not in st.session_state:
        st.session_state.frame_counter = 0
        
    # Track camera mode changes to reset state
    if "previous_camera_mode" not in st.session_state:
        st.session_state.previous_camera_mode = camera_mode
    elif st.session_state.previous_camera_mode != camera_mode:
        # Clear session data when switching modes
        st.session_state.left_angles = []
        st.session_state.right_angles = []
        st.session_state.timestamps = []
        st.session_state.frame_counter = 0
        st.session_state.start_time = time.time()
        st.session_state.last_df = None
        st.session_state.last_left_meter = None
        st.session_state.last_right_meter = None
        if "enable_tracking" in st.session_state:
            st.session_state.enable_tracking = False
        st.session_state.previous_camera_mode = camera_mode
        
    # Session states
    if "left_angles" not in st.session_state:
        st.session_state.left_angles = []
    if "right_angles" not in st.session_state:
        st.session_state.right_angles = []
        st.session_state.right_angles = []
    if "timestamps" not in st.session_state:
        st.session_state.timestamps = []
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    if "last_df" not in st.session_state:
        st.session_state.last_df = None
    if "last_left_meter" not in st.session_state:
        st.session_state.last_left_meter = None
    if "last_right_meter" not in st.session_state:
        st.session_state.last_right_meter = None

    # Main columns layout
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="color:#2c3e50; margin-bottom:1rem;">Live Motion Analysis</h3>
        """, unsafe_allow_html=True)

        # Choose video processor based on camera mode
        if camera_mode == "Camera (for testing/deployment)":
            video_processor = DummyVideoTransformer
        elif camera_mode == "Private Camera (No Tracking)":
            video_processor = PrivateVideoTransformer
        else:
            video_processor = VideoTransformer

        ctx = webrtc_streamer(
            key="stream",
            video_processor_factory=video_processor,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
            media_stream_constraints={
                "video": {
                    "width": {"min": 320, "ideal": 640, "max": 1280},
                    "height": {"min": 240, "ideal": 480, "max": 720},
                    "frameRate": {"min": 5, "ideal": 15, "max": 30}
                } if camera_mode != "Camera (for testing/deployment)" else False,
                "audio": False
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#2c3e50; margin-bottom:1rem;">Left Arm Mobility</h4>
        """, unsafe_allow_html=True)
        meter_left = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#2c3e50; margin-bottom:1rem;">Right Arm Mobility</h4>
        """, unsafe_allow_html=True)
        meter_right = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom section
    st.markdown("""
    <div class="card">
        <h3 style="color:#2c3e50; margin-bottom:1rem;">Progress Over Time</h3>
    """, unsafe_allow_html=True)
    graph_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    # Timer section
    st.markdown("""
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="color:#2c3e50; margin:0;">Session Details</h3>
            <div class="timer" id="timer">00:00</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Add privacy indicator if in private mode
    if camera_mode == "Private Camera (No Tracking)" and not st.session_state.get("enable_tracking", False):
        st.markdown("""
        <div style="background-color: #ecfdf5; border: 1px solid #6ee7b7; color: #065f46; 
                    padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; font-size: 0.9rem;">
            <strong>🔒 Privacy Active:</strong> No motion data is being recorded or stored.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    timer_placeholder = st.empty()

    # Main processing loop with optimized data collection
    if camera_mode == "Camera (for testing/deployment)":
        # For dummy camera, we can run without webrtc context
        if st.button("Start Session"):
            dummy_transformer = DummyVideoTransformer()
            
            # Simulate real-time data collection
            for i in range(100):  # Simulate 100 data points
                left_angle = dummy_transformer.left_angle
                right_angle = dummy_transformer.right_angle
                
                # Simulate frame processing
                dummy_frame = dummy_transformer.transform(None)
                
                st.session_state.left_angles.append(left_angle)
                st.session_state.right_angles.append(right_angle)
                st.session_state.timestamps.append(time.time())
                
                # Update display every 10 iterations
                if i % 10 == 0:
                    buf_left = draw_angle_meter(left_angle, "Left Arm")
                    buf_right = draw_angle_meter(right_angle, "Right Arm")
                    meter_left.image(buf_left)
                    meter_right.image(buf_right)
                    
                    if len(st.session_state.left_angles) > 1:
                        df = pd.DataFrame({
                            "Time": st.session_state.timestamps,
                            "Left Arm Angle": st.session_state.left_angles,
                            "Right Arm Angle": st.session_state.right_angles
                        })
                        df["Relative Time"] = df["Time"] - df["Time"].iloc[0]
                        graph_placeholder.line_chart(df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])
                
                time.sleep(0.1)  # Simulate real-time delay
    
    elif camera_mode == "Private Camera (No Tracking)":
        # For private camera, display live angles but don't log data by default
        if "enable_tracking" not in st.session_state:
            st.session_state.enable_tracking = False
            
        # Add a toggle for enabling tracking in private mode
        enable_tracking = st.checkbox("Enable Data Tracking in Private Mode", 
                                    value=st.session_state.enable_tracking,
                                    help="When checked, your motion data will be tracked and stored for this session only")
        
        if enable_tracking != st.session_state.enable_tracking:
            st.session_state.enable_tracking = enable_tracking
            if enable_tracking:
                st.success("Data tracking enabled. Your motion will now be recorded for this session.")
            else:
                st.warning("Data tracking disabled. Your motion is not being recorded.")
                # Clear existing data
                st.session_state.left_angles = []
                st.session_state.right_angles = []
                st.session_state.timestamps = []
        
        # Update placeholders with "live only" message when tracking is disabled
        if not st.session_state.enable_tracking:
            graph_placeholder.markdown("""
            <div style="background-color: #ecfdf5; border: 1px solid #6ee7b7; color: #065f46; 
                        padding: 2rem; border-radius: 8px; text-align: center; height: 200px; 
                        display: flex; align-items: center; justify-content: center;">
                <div>
                    <h3>🔒 Private Mode Active</h3>
                    <p>Data tracking is disabled. Enable tracking to see your progress graph.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display default angle meters
            default_left_angle = 90
            default_right_angle = 90
            buf_left = draw_angle_meter(default_left_angle, "Left Arm")
            buf_right = draw_angle_meter(default_right_angle, "Right Arm")
            meter_left.image(buf_left)
            meter_right.image(buf_right)
            
        # Private camera processing loop
        while ctx.state.playing:
            if ctx.video_transformer:
                st.session_state.frame_counter += 1
                
                # Get angles from the transformer
                left_angle = ctx.video_transformer.left_angle
                right_angle = ctx.video_transformer.right_angle
                
                # Only if tracking is enabled, we store the data
                if st.session_state.enable_tracking:
                    # Collect data less frequently to reduce memory usage
                    if st.session_state.frame_counter % st.session_state.data_collection_interval == 0:
                        st.session_state.left_angles.append(left_angle)
                        st.session_state.right_angles.append(right_angle)
                        st.session_state.timestamps.append(time.time())

                        # Limit data storage to prevent memory issues
                        if len(st.session_state.left_angles) > 500:
                            st.session_state.left_angles = st.session_state.left_angles[-500:]
                            st.session_state.right_angles = st.session_state.right_angles[-500:]
                            st.session_state.timestamps = st.session_state.timestamps[-500:]
                
                # Always update the angle meters for live display
                if st.session_state.frame_counter % (st.session_state.data_collection_interval * 2) == 0:
                    buf_left = draw_angle_meter(left_angle, "Left Arm")
                    buf_right = draw_angle_meter(right_angle, "Right Arm")
                    meter_left.image(buf_left)
                    meter_right.image(buf_right)
                    
                    # Store last meters if tracking is enabled
                    if st.session_state.enable_tracking:
                        st.session_state.last_left_meter = buf_left
                        st.session_state.last_right_meter = buf_right
                
                # Update chart less frequently (every 10 data points) if tracking is enabled
                if st.session_state.enable_tracking and len(st.session_state.left_angles) % 10 == 0 and len(st.session_state.left_angles) > 0:
                    df = pd.DataFrame({
                        "Time": st.session_state.timestamps,
                        "Left Arm Angle": st.session_state.left_angles,
                        "Right Arm Angle": st.session_state.right_angles
                    })
                    df["Relative Time"] = df["Time"] - df["Time"].iloc[0]
                    graph_placeholder.line_chart(df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])
                    st.session_state.last_df = df
                
                # Update timer every 30 frames
                if st.session_state.frame_counter % 30 == 0:
                    elapsed = int(time.time() - st.session_state.start_time)
                    mins, secs = divmod(elapsed, 60)
                    timer_text = f"⏳ Session Duration: {mins:02d}:{secs:02d}"
                    if not st.session_state.enable_tracking:
                        timer_text += " (not recording)"
                    timer_placeholder.markdown(f'<div class="timer">{timer_text}</div>', unsafe_allow_html=True)
                
            time.sleep(0.033)  # ~30 FPS limit
                
    else:
        # Original real camera processing loop
        while ctx.state.playing:
            if ctx.video_transformer:
                st.session_state.frame_counter += 1
                
                # Collect data less frequently to reduce memory usage
                if st.session_state.frame_counter % st.session_state.data_collection_interval == 0:
                    left_angle = ctx.video_transformer.left_angle
                    right_angle = ctx.video_transformer.right_angle

                    st.session_state.left_angles.append(left_angle)
                    st.session_state.right_angles.append(right_angle)
                    st.session_state.timestamps.append(time.time())

                    # Limit data storage to prevent memory issues (keep last 500 points)
                    if len(st.session_state.left_angles) > 500:
                        st.session_state.left_angles = st.session_state.left_angles[-500:]
                        st.session_state.right_angles = st.session_state.right_angles[-500:]
                        st.session_state.timestamps = st.session_state.timestamps[-500:]

                    # Update meters less frequently
                    if st.session_state.frame_counter % (st.session_state.data_collection_interval * 3) == 0:
                        buf_left = draw_angle_meter(left_angle, "Left Arm")
                        buf_right = draw_angle_meter(right_angle, "Right Arm")
                        meter_left.image(buf_left)
                        meter_right.image(buf_right)

                        st.session_state.last_left_meter = buf_left
                        st.session_state.last_right_meter = buf_right

                    # Update chart less frequently (every 10 data points)
                    if len(st.session_state.left_angles) % 10 == 0:
                        df = pd.DataFrame({
                            "Time": st.session_state.timestamps,
                            "Left Arm Angle": st.session_state.left_angles,
                            "Right Arm Angle": st.session_state.right_angles
                        })
                        df["Relative Time"] = df["Time"] - df["Time"].iloc[0]
                        graph_placeholder.line_chart(df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])
                        st.session_state.last_df = df

                # Update timer every 30 frames
                if st.session_state.frame_counter % 30 == 0:
                    elapsed = int(time.time() - st.session_state.start_time)
                    mins, secs = divmod(elapsed, 60)
                    timer_placeholder.markdown(f"""
                    <div class="timer">
                        ⏳ Session Duration: {mins:02d}:{secs:02d}
                    </div>
                    """, unsafe_allow_html=True)

            time.sleep(0.033)  # ~30 FPS limit

    # After stop - show results if we have data
    if (camera_mode != "Camera (for testing/deployment)" and not ctx.state.playing) or \
       (camera_mode == "Camera (for testing/deployment)" and len(st.session_state.left_angles) > 0):
        if st.session_state.last_df is not None or len(st.session_state.left_angles) > 0:
            # Create final dataframe if not exists
            if st.session_state.last_df is None and len(st.session_state.left_angles) > 0:
                st.session_state.last_df = pd.DataFrame({
                    "Time": st.session_state.timestamps,
                    "Left Arm Angle": st.session_state.left_angles,
                    "Right Arm Angle": st.session_state.right_angles
                })
                st.session_state.last_df["Relative Time"] = st.session_state.last_df["Time"] - st.session_state.last_df["Time"].iloc[0]
            st.markdown("""
            <div class="success-box">
                <h3 style="color:white; margin:0;">✅ Session Completed Successfully!</h3>
                <p style="color:white; margin:0;">Your rehabilitation data has been recorded.</p>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.last_left_meter and st.session_state.last_right_meter:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="color:#2c3e50; margin-bottom:1rem;">Final Left Arm Reading</h4>
                    """, unsafe_allow_html=True)
                    st.image(st.session_state.last_left_meter)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="color:#2c3e50; margin-bottom:1rem;">Final Right Arm Reading</h4>
                    """, unsafe_allow_html=True)
                    st.image(st.session_state.last_right_meter)
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <h3 style="color:#2c3e50; margin-bottom:1rem;">Session Summary</h3>
            """, unsafe_allow_html=True)
            st.line_chart(st.session_state.last_df.set_index("Relative Time")[["Left Arm Angle", "Right Arm Angle"]])
            st.markdown("</div>", unsafe_allow_html=True)

            csv = st.session_state.last_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Session Data (CSV)",
                data=csv,
                file_name="neurotrack_session_data.csv",
                mime="text/csv",
                key="download-csv"
            )


if __name__ == "__main__":
    main()