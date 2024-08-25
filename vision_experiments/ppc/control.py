import numpy as np

class SpeedController:
    """
    The Control Command for throttle, which calculates throttle by increasing speed for 
    lower steering and vice versa, also accounting for a fixed future steering calculated
    by converting a lookahead range into an angle.
    """
    def __init__(self, max_speed=1, min_speed=0.3):
        self.max_speed = max_speed
        self.min_speed = min_speed

    def compute_throttle(self, steering_angle, future_steering):
        # Adjust throttle based on the steering angle
        curvature = abs(steering_angle)
        future_curvature = abs(future_steering)

        # Compute throttle based on current and predicted curvature
        throttle = self.max_speed - curvature * (self.max_speed - self.min_speed)
        throttle -= future_curvature * 0.1  # Additional adjustment factor (alpha)

        return max(self.min_speed, throttle)

def compute_controls(steering_angle, processed_ranges):
    speed_controller = SpeedController(max_speed=1, min_speed=0.3)  # Adjust max_speed as necessary

    # Predict future steering angle based on a window of LIDAR data
    future_steering_angle = predict_future_steering(steering_angle, processed_ranges)

    throttle = speed_controller.compute_throttle(steering_angle, future_steering_angle)
    return throttle, steering_angle

def predict_future_steering(current_steering, lidar_ranges):
    # Look ahead a few points to predict future steering adjustments
    lookahead = 10  # Number of points to look ahead fixed
    angle_increment = 2 * np.pi / len(lidar_ranges)
    
    current_index = int((current_steering + np.pi) / angle_increment)
    future_indices = range(current_index, min(current_index + lookahead, len(lidar_ranges)))
    
    future_angles = [(i * angle_increment - np.pi) / np.pi for i in future_indices if lidar_ranges[i] > 0]
    if future_angles:
        return np.mean(future_angles)
    else:
        return current_steering