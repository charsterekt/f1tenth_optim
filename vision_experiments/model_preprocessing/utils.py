import numpy as np
import torch
from PIL import Image
import torch.nn as nn

def preprocess_data_scaled(vehicle):
    """
    Preprocess the data received from the vehicle's sensors for the model.
    Normalize image data and preprocess LiDAR data. Scaled model dupe function.
    """
    # Normalize the image data
    front_camera_image = vehicle.front_camera_image

    # Preprocess LiDAR data (replace values > 15 with 0)
    lidar_ranges = np.array(vehicle.lidar_range_array)
    lidar_ranges[lidar_ranges == np.inf] = 0

    # Ensure the image is resized to (360, 640) if not already
    if front_camera_image.shape[:2] != (180, 320):
        # Convert the image to PIL format for resizing
        front_camera_image = Image.fromarray((front_camera_image * 255).astype(np.uint8))
        front_camera_image = front_camera_image.resize((320, 180))  # Change the sizing or use the duplicate function
        front_camera_image = np.array(front_camera_image) / 255.0

    # Change the image to channel-first format
    front_camera_image = np.transpose(front_camera_image, (2, 0, 1))

    return front_camera_image, lidar_ranges

def preprocess_data(vehicle):
    """
    Preprocess the data received from the vehicle's sensors for the model.
    Normalize image data and preprocess LiDAR data.
    """
    # Normalize the image data
    front_camera_image = vehicle.front_camera_image

    # Preprocess LiDAR data (replace values > 15 with 0)
    lidar_ranges = np.array(vehicle.lidar_range_array)
    lidar_ranges[lidar_ranges == np.inf] = 0

    # Ensure the image is resized to (360, 640) if not already
    if front_camera_image.shape[:2] != (360, 640):
        # Convert the image to PIL format for resizing
        front_camera_image = Image.fromarray((front_camera_image * 255).astype(np.uint8))
        front_camera_image = front_camera_image.resize((640, 360))
        front_camera_image = np.array(front_camera_image) / 255.0

    # Change the image to channel-first format
    front_camera_image = np.transpose(front_camera_image, (2, 0, 1))

    return front_camera_image, lidar_ranges

def add_to_buffer(image_buffer, lidar_buffer, image, lidar, sequence_length):
    """
    Add data to the buffer and manage its length. For the vision buffer.
    """
    image_buffer.append(image)
    lidar_buffer.append(lidar)
    if len(image_buffer) > sequence_length:
        image_buffer.pop(0)
        lidar_buffer.pop(0)

def is_buffer_full(image_buffer, lidar_buffer, sequence_length):
    """
    Check if the buffer is full. Only process if it is.
    """
    return len(image_buffer) == sequence_length and len(lidar_buffer) == sequence_length

def buffers_to_tensors(image_buffer, lidar_buffer, device):
    """
    Convert buffer contents to tensors for model prediction.
    Ensure the tensors are of type float32 for passing.
    """
    images_seq = np.array(image_buffer).astype(np.float32)
    lidars_seq = np.array(lidar_buffer).astype(np.float32) / 10.0  # Normalizing the lidar data as done during training

    images_seq_tensor = torch.tensor(images_seq).unsqueeze(0).to(device)
    lidars_seq_tensor = torch.tensor(lidars_seq).unsqueeze(0).to(device)
    
    return images_seq_tensor, lidars_seq_tensor


def prepare_model_for_ptq(model):
    """
    Apply the same quantisation conversion to PTQ models for loading them correctly
    using the same hybrid approach as training.
    """
    # Static quantization for the CNN part using the default qconfig for 'qnnpack'
    model.time_distributed_cnn.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    torch.ao.quantization.prepare(model.time_distributed_cnn, inplace=True)

    # Convert the CNN part to a quantized version
    torch.ao.quantization.convert(model.time_distributed_cnn, inplace=True)

    # Dynamic quantization for the LSTM part
    model.lstm_image.qconfig = torch.quantization.default_dynamic_qconfig
    model.lstm_lidar.qconfig = torch.quantization.default_dynamic_qconfig
    model.lstm_image = torch.quantization.quantize_dynamic(model.lstm_image, {nn.LSTM}, dtype=torch.qint8)
    model.lstm_lidar = torch.quantization.quantize_dynamic(model.lstm_lidar, {nn.LSTM}, dtype=torch.qint8)
    return model


def prepare_model_for_qat(model):
    """
    Apply the same quantisation conversion to QAT models for loading them correctly using
    the same approach as training.
    """

    qat_qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    # Apply the QAT configuration to the model
    model.time_distributed_cnn.qconfig = qat_qconfig
    
    # Prepare the model for QAT
    torch.quantization.prepare_qat(model.time_distributed_cnn, inplace=True)

    torch.ao.quantization.convert(model.time_distributed_cnn, inplace=True)
    return model