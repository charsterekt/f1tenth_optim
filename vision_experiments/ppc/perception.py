import numpy as np

def process_data(vehicle, model=False):
    """
    Process the data received from the vehicle's sensors.
    Extract relevant information for planning and control.
    """
    perception_data = {
        'throttle': vehicle.throttle,
        'steering': vehicle.steering,
        'encoder_ticks': vehicle.encoder_ticks,
        'encoder_angles': vehicle.encoder_angles,
        'position': vehicle.position,
        'orientation_quaternion': vehicle.orientation_quaternion,
        'orientation_euler_angles': vehicle.orientation_euler_angles,
        'angular_velocity': vehicle.angular_velocity,
        'linear_acceleration': vehicle.linear_acceleration,
        'lidar_scan_rate': vehicle.lidar_scan_rate,
        'lidar_ranges': vehicle.lidar_range_array,
        'lidar_intensities': vehicle.lidar_intensity_array,
        'front_camera_image': vehicle.front_camera_image
    }
    if model:
        perception_data = {
            'lidar_ranges': vehicle.lidar_range_array,
            'front_camera_image': vehicle.front_camera_image
        }
    return perception_data