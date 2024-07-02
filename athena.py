import socketio
import eventlet
from devkit.adss_toolkit.autodrive_py import autodrive
from ppc import perception, planning, control
from ppc.data_recorder import record_data

from flask import Flask
import numpy as np


# Initialize vehicle 
athena = autodrive.F1TENTH()
athena.id = 'V1'

# Initialize the server
sio = socketio.Server()

# Flask (web) app
app = Flask(__name__)

# On connecting to the sim
@sio.on('connect')
def connect(sid, environ):
    print('Connected!')

# Bridge communicates with the sim
@sio.on('Bridge')
def bridge(sid, data):
    if data:
        """
        PERCEPTION MODULE
        """

        # Directly receive data from the sim
        athena.parse_data(data, verbose=False)
        record_data(athena)
        perception_data = perception.process_data(athena)
        # print(perception_data)

        """
        PLANNING MODULE
        """
        steering_angle, ranges = planning.plan_path(perception_data)
        # print(f"Planned Steering Angle: {steering_angle}")

        """
        CONTROL MODULE
        """
        throttle, steering = control.compute_controls(steering_angle, ranges)

        """
        Publish Computed Control Commands
        Ranges:
        > -1: Full reverse/left
        >  0: No throttle/straight
        >  1: Full forward/right
        """

        athena.throttle_command = throttle
        athena.steering_command = steering

        json_msg = athena.generate_commands(verbose=False) # Publish to the agent

        try:
            sio.emit('Bridge', data=json_msg)
        except Exception as exception_instance:
            print(exception_instance)


# Run the webserver which acts as a node for the agent on the sim
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), app)