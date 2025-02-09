import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import vlc
import time
import os

# Global variables
model = None
counter = 0
fps_list = []  # List to track FPS values for calculating average FPS

# Initialize session state for alarm sound if not already set
if 'alarm_sound' not in st.session_state:
    st.session_state['alarm_sound'] = "Android Notification Sound Effect.wav"  # Default alarm sound

# Load YOLOv5 model function
def load_model():
    global model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/last.pt')

# Function to reset counter
def reset():
    global counter
    counter = 0

# Function to calculate and display results after detection ends
def calculate_metrics():
    avg_fps = np.mean(fps_list) if fps_list else 0
    st.subheader(f"Average FPS (Speed): {avg_fps:.2f} frames/sec")
    st.subheader(f"Total drowsiness detections: {counter}")
    # Placeholder for accuracy, precision, recall (you will need ground truth data)
    st.subheader("Accuracy: N/A (Requires ground truth data)")
    st.subheader("Precision: N/A (Requires ground truth data)")
    st.subheader("Recall: N/A (Requires ground truth data)")

# Function to detect drowsiness
def detect():
    global counter, fps_list
    cap = cv2.VideoCapture(0)
    vid_frame = st.empty()  # Streamlit video frame display
    counter_display = st.empty()  # Counter display
    fps_display = st.empty()  # FPS display

    fps_list = []  # Reset FPS list for new detection session

    st.session_state['stop_detection'] = False  # Reset stop detection state

    while not st.session_state['stop_detection']:
        start_time = time.time()  # Start time for FPS calculation
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Convert frame for model detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        # Display the frame with results
        img = np.squeeze(results.render())  # Render the detection results on the frame
        img_pil = Image.fromarray(img)

        # Update video frame in Streamlit
        vid_frame.image(img_pil, channels="RGB")

        # Check detection results
        if len(results.xywh[0]) > 0:
            dconf = results.xywh[0][0][4]
            dclass = results.xywh[0][0][5]

            # Check if drowsiness is detected
            if dconf.item() > 0.65 and dclass.item() == 16.0:
                # Play sound alert if drowsiness is detected
                p = vlc.MediaPlayer(st.session_state['alarm_sound'])  # Use session state alarm sound
                p.play()
                counter += 1

        # Calculate FPS for this frame
        fps = 1.0 / (time.time() - start_time)
        fps_list.append(fps)

        # Update counter and FPS display
        counter_display.subheader(f"Counter: {counter}")
        fps_display.subheader(f"FPS: {fps:.2f}")

    # Release the video capture when done
    cap.release()
    cv2.destroyAllWindows()

    # Display final metrics once detection ends
    calculate_metrics()

# Set up session state for navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'main'
if 'stop_detection' not in st.session_state:
    st.session_state['stop_detection'] = False

# Main menu function
def main_menu():
    st.title("Hi! This is a Deep Drowsiness Detection Model")
    st.subheader("Drowsiness Detection using YOLOv5")
    
    if st.button("Start Detecting", key="start_detect_button"):
        st.session_state['page'] = 'detect'
        st.session_state['stop_detection'] = False  # Reset stop state
    
    if st.button("Change Alarm", key="change_alarm_button"):
        st.session_state['page'] = 'change_alarm'

# Drowsiness detection page
def detection_page():
    if model is None:
        load_model()  # Load model only when needed
    reset()  # Reset counter before starting detection
    st.button("Go Back to Main Menu", key="back_to_menu_button", on_click=lambda: st.session_state.update({'page': 'main', 'stop_detection': True}))
    detect()  # Run detection function

# Change alarm sound page
def change_alarm_page():
    # Allow user to upload a new alarm sound
    uploaded_file = st.file_uploader("Choose an alarm sound (WAV format)", type=["wav"])
    if uploaded_file is not None:
        alarm_sound_path = os.path.join("alarm_sounds", uploaded_file.name)
        with open(alarm_sound_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['alarm_sound'] = alarm_sound_path  # Update the alarm sound in session state
        st.success(f"Alarm sound changed to: {uploaded_file.name}")

    # Button to go back to the main menu with a unique key
    if st.button("Go Back to Main Menu", key="back_to_menu_from_change_alarm_button"):
        st.session_state['page'] = 'main'

# Navigation logic
if st.session_state['page'] == 'main':
    main_menu()
elif st.session_state['page'] == 'detect':
    detection_page()
elif st.session_state['page'] == 'change_alarm':
    change_alarm_page()