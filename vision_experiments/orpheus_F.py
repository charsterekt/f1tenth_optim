import socketio
import eventlet
from devkit.adss_toolkit.autodrive_py import autodrive
from flask import Flask
import torch
import time
from model_preprocessing.utils import preprocess_data, add_to_buffer, is_buffer_full, buffers_to_tensors
from ppc import perception, planning, control
import numpy as np
from torch_classes.orpheus_F import CNNLSTMModelStudent

# Initialize vehicle 
orpheus_E = autodrive.F1TENTH()
orpheus_E.id = 'V1'

# Initialize the server
sio = socketio.Server()
app = Flask(__name__)

# Load the trained model
model_path = '/Users/chari/autodrive/models/distilled_student_model.pth'
device = torch.device('cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")
model = CNNLSTMModelStudent(seq_length=10)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  
print("Model Live")

# Buffers for storing sequences
image_buffer = []
lidar_buffer = []
sequence_length = 10
total_inference = []

def generate_steering_and_throttle(vehicle):
    """
    Generate steering angle and throttle using the preprocessed data and the trained model.
    Log the inference time for the model at each step.
    """
    # Preprocess data
    front_camera_image, lidar_ranges = preprocess_data(vehicle)
    add_to_buffer(image_buffer, lidar_buffer, front_camera_image, lidar_ranges, sequence_length)

    if not is_buffer_full(image_buffer, lidar_buffer, sequence_length):
        return None, None, None

    images_seq_tensor, lidars_seq_tensor = buffers_to_tensors(image_buffer, lidar_buffer, device)

    # Start timer
    start_time = time.time()

    # Predict steering angle and throttle
    with torch.no_grad():
        predicted_target = model(images_seq_tensor, lidars_seq_tensor).cpu().numpy()

    # End timer
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.4f} seconds")
    
    steering_scaler = 2.5
    throttle_scaler = 0.75

    # Extract and clip the predicted steering angle and throttle
    predicted_steering_angle = np.clip(predicted_target[0, 0], -1, 1) * steering_scaler
    predicted_throttle = np.clip(predicted_target[0, 1], 0, 1) * throttle_scaler

    print(f"Predicted Steering Angle: {predicted_steering_angle}")
    print(f"Predicted Throttle: {predicted_throttle}")

    return predicted_steering_angle, predicted_throttle, inference_time

@sio.on('connect')
def connect(sid, environ):
    print('Connected!')

@sio.on('Bridge')
def bridge(sid, data):
    if data:
        orpheus_E.parse_data(data, verbose=False)
        perception_data = perception.process_data(orpheus_E)

        steering_pred, throttle_pred, inference_time = generate_steering_and_throttle(orpheus_E)
        total_inference.append(inference_time)

        ground_truth_steering, ranges = planning.plan_path(perception_data)
        print(f"Ground Truth Steering: {ground_truth_steering}")

        ground_truth_throttle, _ = control.compute_controls(ground_truth_steering, ranges)
        print(f"Ground Truth Throttle: {ground_truth_throttle}")

        if steering_pred is not None and throttle_pred is not None:
            orpheus_E.throttle_command = throttle_pred
            orpheus_E.steering_command = steering_pred
        else:
            orpheus_E.throttle_command = 0
            orpheus_E.steering_command = 0

        json_msg = orpheus_E.generate_commands(verbose=False)
        try:
            sio.emit('Bridge', data=json_msg)
        except Exception as e:
            print(e)

@sio.on('disconnect')
def disconnect(sid):
    print('Disconnected!')
    # print(total_inference[9:])
    print(f"Average Inference Time: {np.average(total_inference[9:]):.4f} for {len(total_inference[9:])} predictions")

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), app)