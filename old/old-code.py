import cv2
import numpy as np
import time

# Function to extract frames from video
def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    while success:
        frames.append(frame)
        success, frame = video.read()
    video.release()
    return frames

# Function to calculate optical flow between two frames
def calculate_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Function to calculate speed based on flow
def estimate_speed(flow, time_interval, scale_factor):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_magnitude = np.mean(mag)
    speed_m_per_s = (avg_magnitude / time_interval) * scale_factor  # Speed in meters per second
    speed_km_per_h = speed_m_per_s * 3.6  # Speed in kilometers per hour
    return speed_m_per_s, speed_km_per_h

# Function to detect vehicles using the Haar Cascade model
def detect_vehicle(frame, cascade_path):
    car_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vehicles = car_cascade.detectMultiScale(gray, 1.1, 1)
    return vehicles

# Main function for vehicle speed estimation
def main(video_path, cascade_path, time_interval=1.0, scale_factor=0.1):
    frames = extract_frames(video_path)

    prev_frame = frames[0]
    for i in range(1, len(frames)):
        next_frame = frames[i]
        
        # Detect vehicle (for simplicity, using first vehicle detected)
        vehicles = detect_vehicle(prev_frame, cascade_path)
        if len(vehicles) > 0:
            x, y, w, h = vehicles[0]  # First detected vehicle
            cv2.rectangle(prev_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('Vehicle Detection', prev_frame)
            cv2.waitKey(10)

        # Calculate optical flow between frames
        flow = calculate_optical_flow(prev_frame, next_frame)

        # Estimate speed of vehicle
        speed_m_per_s, speed_km_per_h = estimate_speed(flow, time_interval, scale_factor)
        print(f"Estimated speed: {speed_m_per_s:.2f} m/s, {speed_km_per_h:.2f} km/h")

        # Update previous frame
        prev_frame = next_frame

        # Introduce delay for real-time simulation
        time.sleep(time_interval)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "File0010.mp4"  # Replace with your video path
    cascade_path = "car.xml"  # Replace with the path to your Haar Cascade file (or 'cars.xml')
    main(video_path, cascade_path)
