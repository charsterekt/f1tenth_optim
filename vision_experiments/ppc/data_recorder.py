import pandas as pd
import cv2
import os
import logging
import threading

recording = False
data_records = []
frame_count = 0
frame_dir = "recorded_frames_new"
lock = threading.Lock()

if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

def start_recording():
    global recording, data_records, frame_count
    with lock:
        recording = True
        data_records = []
        frame_count = 0
    logging.info("Recording started")

def stop_recording():
    global recording
    with lock:
        recording = False
    logging.info("Recording stopped")

def save_data():
    global data_records, frame_count
    with lock:
        # For each data record, a corresponding frame should exist
        valid_records = [record for record in data_records if os.path.exists(record['camera_frame'])]
        df = pd.DataFrame(valid_records)
        df.to_csv('recorded_data_new.csv', index=False)
    logging.info(f"Saved {len(valid_records)} frames and data to recorded_data.csv and {frame_dir}.")

def record_data(vehicle):
    global frame_count, recording, data_records
    if recording:
        frame_path = os.path.join(frame_dir, f"{frame_count}.jpg")
        cv2.imwrite(frame_path, vehicle.front_camera_image)
        if os.path.exists(frame_path):
            data_record = {
                'timestamp': pd.Timestamp.now(),
                'throttle': vehicle.throttle,
                'steering': vehicle.steering,
                'encoder_ticks_left': vehicle.encoder_ticks[0],
                'encoder_ticks_right': vehicle.encoder_ticks[1],
                'encoder_angles_left': vehicle.encoder_angles[0],
                'encoder_angles_right': vehicle.encoder_angles[1],
                'position_x': vehicle.position[0],
                'position_y': vehicle.position[1],
                'position_z': vehicle.position[2],
                'orientation_quaternion_x': vehicle.orientation_quaternion[0],
                'orientation_quaternion_y': vehicle.orientation_quaternion[1],
                'orientation_quaternion_z': vehicle.orientation_quaternion[2],
                'orientation_quaternion_w': vehicle.orientation_quaternion[3],
                'orientation_euler_roll': vehicle.orientation_euler_angles[0],
                'orientation_euler_pitch': vehicle.orientation_euler_angles[1],
                'orientation_euler_yaw': vehicle.orientation_euler_angles[2],
                'angular_velocity_x': vehicle.angular_velocity[0],
                'angular_velocity_y': vehicle.angular_velocity[1],
                'angular_velocity_z': vehicle.angular_velocity[2],
                'linear_acceleration_x': vehicle.linear_acceleration[0],
                'linear_acceleration_y': vehicle.linear_acceleration[1],
                'linear_acceleration_z': vehicle.linear_acceleration[2],
                'lidar_scan_rate': vehicle.lidar_scan_rate,
                'lidar_ranges': vehicle.lidar_range_array.tolist(),
                'camera_frame': frame_path
            }
            with lock:
                data_records.append(data_record)
                frame_count += 1

def hotkey_listener():
    from pynput import keyboard

    def on_press(key):
        try:
            if key.char == 's':
                if recording:
                    print("RECORDING STOPPED")
                    stop_recording()
                    save_data()
                else:
                    print("RECORDING STARTED")
                    start_recording()
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# Start hotkey listener thread
threading.Thread(target=hotkey_listener, daemon=True).start()