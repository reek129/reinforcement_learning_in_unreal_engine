import os,subprocess,time
import airsim
import importlib,json
import numpy as np
import math

import nvidia_smi
import psutil
from os import getpid
import pygame
import sys
import random
import cv2
from PIL import Image

import tempfile
import pprint

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import tkinter as tk
from tkinter import scrolledtext,simpledialog


from datetime import datetime

import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#    drone related access and controll
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print(axes, "failed")

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az



def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


def get_MonocularImageRGB(client, vehicle_name):

    responses1 = client.simGetImages([
        airsim.ImageRequest('front_center', airsim.ImageType.Scene, False,
                            False)], vehicle_name=vehicle_name)  # scene vision image in uncompressed RGBA array

    response = responses1[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_rgba = img1d.reshape(response.height, response.width, 3)
    img = Image.fromarray(img_rgba)
    img_rgb = img.convert('RGB')
    camera_image_rgb = np.asarray(img_rgb)
    camera_image = camera_image_rgb

    return camera_image

# DeepQLearning Network


# #######################################



# agent[name_agent] = DroneAgent(algorithm_cfg, client, name='DQN', vehicle_name=name_agent)
class DroneAgent():
    def __init__(self, cfg, client, name, vehicle_name):
        print(cfg)
        self.cfg = cfg
        # self.env_type = cfg['env_type']
        self.input_size = cfg['input_size']
        self.num_actions = cfg['num_actions']
        self.iter = 0
        self.vehicle_name = vehicle_name
        self.client = client
        self.lidar_max_range = 50  # Maximum LiDAR range in meters
        self.image_size = (64, 64)  # Resize images to a fixed size
        self.state_dim = cfg['input_size'] 

        self.crash_penalty = -100  # Penalty for collision
        self.step_penalty = -0.1  # Small penalty for each step to encourage efficiency
        self.goal_reward = 100  # Reward for reaching the goal
        self.boundary_penalty = -50  # Penalty for going out of bounds
        self.safe_distance = 5  # Minimum safe distance from obstacles (meters)

        # RL-specific parameters
        self.epsilon = 1.0  # Initial exploration rate (100% exploration)
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.gamma = cfg.get("gamma", 0.99)  # Discount factor
        self.learning_rate = cfg.get("learning_rate", 1e-3)

        # Replay buffer for storing experiences
        self.experience_replay = []  # Initialize as an empty list
        self.buffer_limit = cfg.get("buffer_len", 10000)  # Max size of the replay buffer
        self.batch_size = cfg.get("batch_size", 32)  # Batch size for training

        # Neural networks for Q-learning
        # self.q_network = self.initialize_network()
        # self.target_network = self.initialize_network()

        self.q_network = InitializeNetworkDeepQLearning(cfg, name, vehicle_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.target_network = InitializeNetworkDeepQLearning(cfg, name, vehicle_name).to("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)



        print('Initializing from drone agent ', self.vehicle_name )
        # write code to get a network model can be CNN, diffusion or something else
        # self.network_model = initialize_network_DeepQLearning(cfg,name,vehicle_name)

        # sys.exit()

    def set_reset_drone(self):
        self.client.reset()
    
    def land_drone(self):
        self.client.landAsync().join()

    def move_to_initial_position(self, initial_position):
        """
        Moves the drone back to its initial position.
        Args:
            initial_position (airsim.Vector3r): The initial position to reset the drone to.
        """
        print(f"Resetting to initial position: {initial_position}")
        self.client.moveToPositionAsync(
            initial_position.x_val,
            initial_position.y_val,
            initial_position.z_val,
            velocity=1,
            vehicle_name=self.vehicle_name
        ).join()

        print(f"Resetting to initial position: {initial_position}")
        self.client.reset()  # Reset the drone environment
        time.sleep(1)  # Allow some time for the environment to stabilize

        # Re-enable API control and arm the drone
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        print("Drone successfully reset and connected.")


    def get_reward2(self, collision_info, current_position=None, goal_position=None, lidar_data=None):
        """
        Calculate the reward for the current step based on the drone's state.
        """
        reward = 0

        # Penalize collisions
        if collision_info.has_collided:
            reward -= 100
            return reward

        # Reward reaching the goal
        if goal_position and current_position:
            distance_to_goal = np.linalg.norm([
                current_position.x_val - goal_position.x_val,
                current_position.y_val - goal_position.y_val,
                current_position.z_val - goal_position.z_val
            ])
            if distance_to_goal < 1.0:
                reward += 100
                return reward
            reward += max(0, 100 / distance_to_goal)

        # Penalty for getting too close to obstacles
        if lidar_data is not None:
            min_distance = np.min(lidar_data) if len(lidar_data) > 0 else 5.0
            if min_distance < 5.0:
                reward -= (5.0 - min_distance)

        # Small penalty for each step
        reward -= 0.1

        return reward

    def get_reward(self, collision_info,initial_position, current_position=None,
                    goal_position=None, lidar_data=None):

        """
        Calculate the reward for the current step based on the drone's state.

        Args:
            collision_info (airsim.CollisionInfo): Information about collisions.
            current_position (airsim.Vector3r): Current position of the drone.
            goal_position (airsim.Vector3r): Goal position for the drone (if applicable).
            lidar_data (np.ndarray): LiDAR distances (optional, for obstacle detection).

        Returns:
            float: The reward for the current step.
        """
        reward = 0

        # 1. Penalize collisions
        if collision_info.has_collided:
            reward += self.crash_penalty
            print("Collision detected! Resetting to initial position.")
            self.move_to_initial_position(initial_position)
            print(f"Collision detected! Applying penalty of {self.crash_penalty}.")
            return reward  # Return immediately since collision is critical

        # 2. Reward reaching the goal
        if goal_position is not None and current_position is not None:
            distance_to_goal = np.linalg.norm([
                current_position.x_val - goal_position.x_val,
                current_position.y_val - goal_position.y_val,
                current_position.z_val - goal_position.z_val
            ])
            if distance_to_goal < 1.0:  # Goal is within 1 meter
                reward += self.goal_reward
                print(f"Goal reached! Applying reward of {self.goal_reward}.")
                return reward

            # Encourage moving closer to the goal
            reward += max(0, self.goal_reward / 10 - distance_to_goal)

        # 3. Penalize going out of bounds
        if current_position is not None:
            if abs(current_position.x_val) > 50 or abs(current_position.y_val) > 50 or abs(current_position.z_val) > 30:
                reward += self.boundary_penalty
                print(f"Out of bounds! Applying penalty of {self.boundary_penalty}.")
                return reward

        # 4. Penalize getting too close to obstacles
        if lidar_data is not None:
            min_distance = np.min(lidar_data) if len(lidar_data) > 0 else self.safe_distance
            if min_distance < self.safe_distance:
                reward += self.step_penalty * (self.safe_distance - min_distance)
                print(f"Obstacle too close! Applying penalty.")

        # 5. Small penalty for every step to encourage efficiency
        reward += self.step_penalty



        return reward


    
    
    def create_client(self):
        self.client = airsim.MultirotorClient(ip=config['ip_address'], timeout_value=10)
        self.client.confirmConnection()
        # return self.get_client()
    
    def set_api_control(self,flag=True):
        self.client.enableApiControl(flag)

    def set_arm_disarm(self,flag=True):
        self.client.armDisarm(flag)

    def set_to_new_pose(self,new_pose,teleport=True):
        success = self.client.simSetObjectPose(config['name_agent'],new_pose,teleport=teleport)
        return success
    
    def set_take_off(self):
        self.client.takeoffAsync(vehicle_name=config["name_agent"])

    def get_new_position(self):
        x = int(input("x_val: "))
        y = int(input("y_val: "))
        z = int(input("z_val: "))

        position ={
            'x':x,'y':y,'z':z
        }
        return position

    def get_new_orientation(self):
        Rx = int(input("Rx_val: "))
        Ry = int(input("Ry_val: "))
        Rz = int(input("Rz_val: "))

        orientation ={
            'x':Rx,'y':Ry,'z':Rz
        }
        return orientation

    def check_if_moved(self,old_pos):
        print("Inside check if moved")
        new_pos = self.get_position_of_drone()
        print(new_pos)
        if abs(old_pos.x_val - new_pos.x_val) > 0 or abs(old_pos.y_val - new_pos.y_val) > 0 or abs(old_pos.z_val - new_pos.z_val) > 0:
            print("Movement successful")
            return True
        else:
            print("Movement failed")
            return False

    def move_drone_by_position(self):
        current_state = self.get_client_state()
        current_position = current_state.kinematics_estimated.position
        current_yaw = airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)[2] * 57.2958  # Convert radians to degrees
        print(f"Current_position: {current_position}")
        print(f"Current yaw: {current_yaw}")

        d_position= self.get_new_position().values()
        # print(d_position)
        dx,dy,dz = d_position

        new_x = current_position.x_val + dx
        new_y = current_position.y_val + dy
        new_z = current_position.z_val + dz

        self.move_by_position(new_x,new_y,new_z)
        success = self.check_if_moved(current_position)
        print(f"Moving drone2: {success}")

    def move_by_velocity(self, vx, vy, vz, duration):
        """
        Move the drone using specified velocities in x, y, z directions.
        Args:
            vx (float): Velocity in the x direction.
            vy (float): Velocity in the y direction.
            vz (float): Velocity in the z direction.
            duration (float): Duration for which the drone should move.
        Returns:
            bool: True if the command is successful, False otherwise.
        """
        try:
            print(f"Moving drone with velocity vx={vx}, vy={vy}, vz={vz} for duration={duration}s")
            self.client.moveByVelocityAsync(vx, vy, vz, duration, vehicle_name=self.vehicle_name).join()
            print("Drone movement by velocity completed.")
            return True
        except Exception as e:
            print(f"Error during velocity movement: {e}")
            return False

    def move_drone_by_position2(self,new_position):
        current_state = self.get_client_state()
        current_position = current_state.kinematics_estimated.position
        current_yaw = airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)[2] * 57.2958  # Convert radians to degrees
        print(f"Current_position: {current_position}")
        print(f"Current yaw: {current_yaw}")

        d_position= new_position
        print(f"new_position: {new_position}")
        # print(d_position)
        # dx,dy,dz = d_position

        # print(type(current_position.x_val))
        # print(type(dx))

        # new_x = current_position.x_val + float(dx)
        # new_y = current_position.y_val + float(dy)
        # new_z = current_position.z_val + float(dz)

        new_x = current_position.x_val + float(d_position['x'])
        new_y = current_position.y_val + float(d_position['y'])
        new_z = current_position.z_val + float(d_position['z'])

        print(new_x,new_y,new_z)

        self.move_by_position(new_x,new_y,new_z)
        success = self.check_if_moved(current_position)
        print(f"Moving drone2: {success}")
        return success
    
    
    def move_by_position(self,new_x, new_y, new_z):
        print("Inside move by position")
        return self.client.moveToPositionAsync(new_x, new_y, new_z, velocity=1).join()

    def hover_drone(self):
        self.client.hoverAsync().join()

   
    def get_client(self):
        return self.client
    
    def get_position_of_drone(self):
        current_pose = self.client.simGetVehiclePose().position
        print(current_pose)
        return current_pose
    
    def get_client_state(self):
        current_state = self.client.getMultirotorState()
        # print(current_state)
        return current_state
    

    
    def get_new_position_cord(self,position):
        return airsim.Vector3r(*position.values())
    
    def get_new_orientation(self,orientation):
        return airsim.to_quaternion(*orientation.values())
    
    def get_new_pose(self,position,orientation):
        new_pose = airsim.Pose(self.get_new_position_cord(position=position),self.get_new_orientation(orientation=orientation))
        return new_pose

    def get_client(self):
        return self.client
    # cxaacasdvs

    def get_api_control_status(self):
        flag = self.client.isApiControlEnabled()
        if flag:
            print("API control is successfully enabled.")
        else:
            print("API control is not enabled.")
        
        return flag
    
    def get_client_ping(self):
        flag = self.client.ping()
        if flag:
            print("Connection to simulator is active.")
        else:
            print("Failed to connect to the simulator.")

        return flag
    
    def get_barometer_data(self,vehicle_name):
        barometer_data = self.client.getBarometerData(vehicle_name=vehicle_name)
        print(barometer_data)
        return barometer_data

    def get_imu_data(self,vehicle_name):
        imu_data = self.client.getImuData(vehicle_name = vehicle_name)
        print(imu_data)
        return imu_data

    def get_gps_data(self,vehicle_name):
        gps_data = self.client.getGpsData(vehicle_name = vehicle_name)
        print(gps_data)
        return gps_data

    def get_magnetometer_data(self,vehicle_name):
        magnetometer_data = self.client.getMagnetometerData(vehicle_name = vehicle_name)
        print(magnetometer_data)
        return magnetometer_data
    
    def get_drone_images(self):
        responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
                airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
                airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
                airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
        
        print('Retrieved images: %d' % len(responses))

        tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
        print ("Saving images to %s" % tmp_dir)
        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise

        for idx, response in enumerate(responses):

            timestamp = "__"+datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

            filename = os.path.join(tmp_dir, str(idx)+timestamp)

            if response.pixels_as_float:
                print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            elif response.compress: #png format
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            else: #uncompressed array
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) # get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

        return tmp_dir
    
    def get_sensor_data(self):
        """
        Aggregate, clean, and preprocess sensor data into a feature vector.
        Returns:
            np.ndarray: The preprocessed feature vector.
        """
        # Initialize empty feature vector
        features = []

        ### 1. Collect Barometer Data
        barometer_data = self.client.getBarometerData(vehicle_name=self.vehicle_name)
        altitude = barometer_data.altitude if barometer_data else 0.0
        pressure = barometer_data.pressure if barometer_data else 0.0
        features.extend([altitude, pressure])  # Append altitude and pressure

        ### 2. Collect IMU Data
        imu_data = self.client.getImuData(vehicle_name=self.vehicle_name)
        imu_acc = imu_data.linear_acceleration if imu_data else airsim.Vector3r(0.0, 0.0, 0.0)
        imu_ang_vel = imu_data.angular_velocity if imu_data else airsim.Vector3r(0.0, 0.0, 0.0)
        features.extend([imu_acc.x_val, imu_acc.y_val, imu_acc.z_val])  # Linear acceleration
        features.extend([imu_ang_vel.x_val, imu_ang_vel.y_val, imu_ang_vel.z_val])  # Angular velocity

        ### 3. Collect GPS Data
        gps_data = self.client.getGpsData(vehicle_name=self.vehicle_name)
        gps_position = gps_data.gnss.geo_point if gps_data else airsim.GeoPoint(0.0, 0.0, 0.0)
        features.extend([gps_position.latitude, gps_position.longitude, gps_position.altitude])

        ### 4. Collect Magnetometer Data
        magnetometer_data = self.client.getMagnetometerData(vehicle_name=self.vehicle_name)
        mag_field = magnetometer_data.magnetic_field_body if magnetometer_data else airsim.Vector3r(0.0, 0.0, 0.0)
        features.extend([mag_field.x_val, mag_field.y_val, mag_field.z_val])

        ### 5. Collect LiDAR Data
        lidar_data = self.client.getLidarData(vehicle_name=self.vehicle_name)
        if lidar_data and len(lidar_data.point_cloud) >= 3:
            lidar_points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            lidar_distances = np.linalg.norm(lidar_points, axis=1)  # Compute distances
            lidar_distances = np.clip(lidar_distances / self.lidar_max_range, 0, 1)  # Normalize
        else:
            lidar_distances = np.zeros(16)  # Placeholder for 16 LiDAR beams
        features.extend(lidar_distances)

        ### 6. Process Image Data
        # Get the front camera RGB image
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.vehicle_name)

        if responses and len(responses) > 0 and responses[0].image_data_uint8:
            # Convert raw image data to a numpy array
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

            # Resize and normalize the image
            img_resized = cv2.resize(img_rgb, self.image_size)  # Resize to (64, 64)
            img_normalized = img_resized / 255.0  # Normalize pixel values to [0, 1]
        else:
            # Placeholder for empty image data
            img_normalized = np.zeros((*self.image_size, 3), dtype=np.float32)

        # Flatten the image array and add to the feature vector
        features.extend(img_normalized.flatten())

        ### 7. Combine and Return Features
        # Convert the feature list to a numpy array
        state_vector = np.array(features, dtype=np.float32)
        return state_vector


    
    def display_point_cloud(self, points):

        tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
        print ("Saving images to %s" % tmp_dir)
        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise

        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        filename = os.path.join(tmp_dir, f"pointcloud_{timestamp}.png")
    

        # Convert points to a NumPy array
        points = np.array(points)
        
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='yellow', s=1)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('LiDAR Point Cloud')

            # Save the plot as a PNG file
        plt.savefig(filename)
        print(f"Point cloud image saved as {filename}")

        # Show plot
        plt.show()
        return tmp_dir
    
    def get_lidar_data(self,vehicle_name):
        print('Scanning Has Started\n')
        print('Use Keyboard Interrupt \'CTRL + C\' to Stop Scanning\n')
        existing_data_cleared = True

        # Storage for point cloud data
        all_points = []  


        try:
            while True:
                for lidar_name in config['lidar_names']:
                    filename = f"{vehicle_name}_{lidar_name}_pointcloud.asc"
                    if not existing_data_cleared:
                        f = open(filename,'w')
                    else:
                        f = open(filename,'a')
                    lidar_data = self.client.getLidarData(lidar_name=lidar_name,vehicle_name=vehicle_name)
                    
                    orientation = lidar_data.pose.orientation
                    q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val

                    # rotation matrix
                    rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                                  [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                                  [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)]))

                    position = lidar_data.pose.position
                    for i in range(0, len(lidar_data.point_cloud), 3):
                        xyz = lidar_data.point_cloud[i:i+3]

                        corrected_x, corrected_y, corrected_z = np.matmul(rotation_matrix, np.asarray(xyz))
                        final_x = corrected_x + position.x_val
                        final_y = corrected_y + position.y_val
                        final_z = corrected_z + position.z_val

                        f.write("%f %f %f %d %d %d \n" % (final_x,final_y,final_z,255,255,0))

                        # Store points for visualization
                        all_points.append([final_x, final_y, final_z])

                    f.close()
                # Plot the point cloud
                self.display_point_cloud(all_points)
                existing_data_cleared = True
        except KeyboardInterrupt:
            airsim.wait_key('Press any key to stop running this script')
            print("Done!\n")
    
    def get_agent_state(self):
        return self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
    
    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore: random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert to tensor
            print(f"Input shape: {state_tensor.shape}")
            print(self.q_network)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # Exploit: best action

    def store_experience(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        if len(self.experience_replay) >= self.buffer_limit:
            self.experience_replay.pop(0)  # Remove oldest experience
        self.experience_replay.append((state, action, reward, next_state, done))

    def train(self):
        """Train the Q-network using a batch of experiences."""
        if len(self.experience_replay) < self.batch_size:
            return  # Skip training if not enough data

        # Sample a batch of experiences
        batch = random.sample(self.experience_replay, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        print(f"Type of states: {type(states)}")
        print(f"Length of states: {len(states)}")
        print(f"Shapes of individual elements in states: {[np.array(s).shape for s in states]}")

        max_shape = max([np.array(s).shape for s in states])
        print(f"max_shape: {max_shape}")
        padded_states = [
            np.pad(s, [(0, max_shape[i] - s.shape[i]) for i in range(len(s.shape))], mode='constant') if s.shape != max_shape else s
            for s in states
        ]

        states = torch.FloatTensor(np.array(padded_states))

        # states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # exit()

        


        # Compute Q-values and target Q-values
        q_values = self.q_network(states).gather(1, actions).squeeze()
        with torch.no_grad():
            target_q_values = rewards + self.gamma * torch.max(self.target_network(next_states), dim=1)[0] * (1 - dones)

        # Compute loss and update the Q-network
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Update the target Q-network with the Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def initialize_network(self):
        """
        Initialize the Q-network. Replace this with your neural network model.
        """
        # Example: Simple feedforward network for illustration
        return nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )












    # end of drone related access and control



# end of DeepQ network models


config ={
    'mode':'train',
    'drone':'DJIPhantom',
    'ip_address':"127.0.0.2",
    'ClockSpeed':20,
    'camera_params' :{
        'width':320,
        'height':180,
        'fov_degrees':80
    },
    'SimMode':'Multirotor',
    "VehicleType": "SimpleFlight",
    
    "SubWindows":[
        {
            "WindowID": 0,
            "ImageType":0,
            "CameraName": "3",
            "Visible":True
         },
        {
            "WindowID": 1,
            "ImageType":3,
            "CameraName": "0",
            "Visible":True
        },
        {
            "WindowID": 2,
            "ImageType":5,
            "CameraName": "4",
            "Visible":True
        }
    ],
    'num_agents':1,
    'env_name':"MyProject2_4point_27",
    # "EnableCollisionPassthrogh": True,
    # "EnableCollisions": False,
    'NVIDIA_GPU':True,
    'name_agent':'drone0',
    'lidar_names': ["LidarSensor1","LidarSensor2"]

}


def MyProject2_4point_27():
    orig_ip = [
        [275.0, 0.0, 144.0]  # Player start location in Unreal (X, Y, Z)
    ]
    rot_ip = {
        "Pitch":0.0,
        "Roll":0.0,
        "Yaw":179.9
    }
    level_name = ['PlayerStart']  # You can name the level or scenario
    crash_threshold = 0.07  # Keeping the same crash threshold as before
    return orig_ip,rot_ip, level_name, crash_threshold

def initial_positions(env_name):
    print(env_name)
    name = env_name+'()'
    orig_ip,rot_ip, level_name, crash_threshold = eval(name)

    # Reset arrays for one drone
    reset_array = []
    reset_array_raw = []
    level_names = []

    # Use the physical player start for raw reference
    physical_player_start = orig_ip[0]

    # Absolute positions (X, Y, Z) in meters
    x1 = orig_ip[0][0] / 100  # Convert cm to meters
    y1 = orig_ip[0][1] / 100  # Convert cm to meters
    z1 = orig_ip[0][2] / 100 

    x_raw = (orig_ip[0][0] - physical_player_start[0]) / 100
    y_raw = (orig_ip[0][1] - physical_player_start[1]) / 100

    # Convert rotation details to radians
    pitch = rot_ip["Pitch"]  * np.pi / 180 
    roll = rot_ip["Roll"]   * np.pi / 180 
    yaw = rot_ip["Yaw"] * np.pi / 180 

    pp = airsim.Pose(airsim.Vector3r(x1, y1, z1), airsim.to_quaternion(pitch, roll, yaw))
    # print(pp)
    reset_array.append(pp)

    # for later debugging
    reset_array_raw.append([x_raw, y_raw, z1, pitch * 180 / np.pi, roll * 180 / np.pi, yaw * 180 / np.pi])
    # print(reset_array_raw)

    return reset_array, reset_array_raw, level_names, crash_threshold

def calculate_rotational_angles(drone_orientation):
    w = drone_orientation.w_val
    x = drone_orientation.x_val
    y = drone_orientation.y_val
    z = drone_orientation.z_val

    # Calculate roll (x-axis rotation)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Calculate pitch (y-axis rotation)
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))  # Clamp the value within [-1, 1]

    # Calculate yaw (z-axis rotation)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # Convert from radians to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)

    # Output the results
    print(f"Roll: {roll_deg:.2f} degrees")
    print(f"Pitch: {pitch_deg:.2f} degrees")
    print(f"Yaw: {yaw_deg:.2f} degrees")

    rot_ip = {
        "Pitch":pitch_deg,
        "Roll":roll_deg,
        "Yaw": yaw_deg
    }
    return rot_ip




def generate_json(config,initZ=0):
    print(config.keys())
    flag  = True
    path = os.path.expanduser('~\Documents\Airsim')
    if not os.path.exists(path):
        os.makedirs(path)

    filename = path + '\settings.json'

    data = {}

    data['SettingsVersion'] = 1.8
    data['LocalHostIp'] = config['ip_address']
    data['SimMode'] = config['SimMode']
    data['ClockSpeed'] = config['ClockSpeed']

    # print(len(config["SubWindows"]))
    data["SubWindows"] = config["SubWindows"]

    # define flight agents
    _, reset_array_raw, _, _ = initial_positions(config['env_name'])
    print(reset_array_raw[0])
    airsim_pose_obj = reset_array_raw[0]

    drone_position = airsim_pose_obj[0:3]
    # drone_orientation = airsim_pose_obj[3:]
    # print(drone_position,type(reset_array))
    # print(drone_orientation)
    # rot_ip = calculate_rotational_angles(drone_orientation)
    Vehicles = {}
    name_agent =config['name_agent']
    Vehicles[name_agent] = {}
    
    Vehicles[name_agent]["VehicleType"] = config["VehicleType"]
    # X,Y,Z,Pitch
    Vehicles[name_agent]["X"] = drone_position[0]
    Vehicles[name_agent]["Y"] = drone_position[1]
    # Vehicles[name_agent]["Z"] = initZ if initZ != 0 else drone_position[2]
    Vehicles[name_agent]["Z"] = 0

    # for key in rot_ip.keys():
    #     # print(key)
    #     Vehicles[name_agent][key]=rot_ip[key]

    
    # adding extra sensors to done (default sensors for drone (IMU,Magnetometer,GPS,Barometer))
    # adding lidar

    sensors = {}

    lidar = {}
    lidar['SensorType'] = 6
    lidar['Enabled']=True
    lidar['NumberOfChannels']=16
    lidar["Roll"]= 0
    lidar["Pitch"]= 90 
    lidar["Yaw"] = 0
    lidar["DrawDebugPoints"] = True
    lidar['DataFrame'] = 'SensorLocalFrame'
    

    sensors[config['lidar_names'][0]]=lidar

    lidar = {}
    lidar['SensorType'] = 6
    lidar['Enabled']=True
    lidar['NumberOfChannels']=16
    lidar["Roll"]= 90
    lidar["Pitch"]= 90 
    lidar["Yaw"] = 0
    lidar["DrawDebugPoints"] = True
    lidar['DataFrame'] = 'SensorLocalFrame'

    sensors[config['lidar_names'][1]]=lidar

    Vehicles[name_agent]["Sensors"]=sensors


    data["Vehicles"] = Vehicles

    CameraDefaults = {}
    CameraDefaults['CaptureSettings']=[]

    camera = {}
    camera['ImageType'] = 0
    camera['Width'] = config['camera_params']['width']
    camera['Height'] = config['camera_params']['height']
    camera['FOV_Degrees'] = config['camera_params']['fov_degrees'] 

    CameraDefaults['CaptureSettings'].append(camera)

    camera = {}
    camera['ImageType'] = 3
    camera['Width'] = config['camera_params']['width']
    camera['Height'] = config['camera_params']['height']
    camera['FOV_Degrees'] = config['camera_params']['fov_degrees']

    CameraDefaults['CaptureSettings'].append(camera)

    camera = {}
    camera['ImageType'] = 5
    camera['Width'] = config['camera_params']['width']
    camera['Height'] = config['camera_params']['height']
    camera['FOV_Degrees'] = config['camera_params']['fov_degrees'] 

    CameraDefaults['CaptureSettings'].append(camera)

    data['CameraDefaults'] = CameraDefaults

    # "EnableCollisionPassthrogh": true,
    # "EnableCollisions": false

    # data['EnableCollisionPassthrogh'] = config['EnableCollisionPassthrogh']
    # data['EnableCollisions'] = config['EnableCollisions']




    print(data)

    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    return flag

def print_orderly(str, n):
    print('')
    hyphens = '-' * int((n - len(str)) / 2)
    print(hyphens + ' ' + str + ' ' + hyphens)

def start_environment(env_name):
    print_orderly('Environment', 80)
    env_folder = os.path.dirname(os.path.abspath(__file__)) + "/unreal_envs/" + env_name + "/"
    path = env_folder + env_name + ".exe"
    # env_process = []
    print(path)
    env_process = subprocess.Popen(path)
    time.sleep(5)
    print("Successfully loaded environment: " + env_name)

    return env_process, env_folder



def connect_drone(ip_address='127.0.0.1', phase='infer'):
    print_orderly('connectDrone', 80)
    client = airsim.MultirotorClient(ip=ip_address, timeout_value=10)
    client.confirmConnection()
    time.sleep(1)

    old_posit = {}
    client.enableApiControl(True, config['name_agent'])
    client.armDisarm(True, config['name_agent'])
    client.takeoffAsync(vehicle_name=config['name_agent'])

    time.sleep(1)
    old_posit[config['name_agent']] = client.simGetVehiclePose(vehicle_name=config['name_agent'])
    print(f"old_posit:{old_posit}")
    initZ = old_posit[config['name_agent']].position.z_val

    return client, old_posit, initZ


deep_q_learning_config={
    'input_size':103,
    'num_actions':25,
    'train_type':'e2e',
    'wait_before_train':5000,
    'max_iters':150000,
    'buffer_len':10000,
    'batch_size': 32,
    'epsilon_saturation':100000,
    'crash_thresh': 1.3,
    'Q_clip': True,
    'train_interval': 2,
    'update_target_interval': 8000,
    'gamma':  0.99,
    'dropout_rate': 0.1,
    'learning_rate':2e-6,
    'switch_env_steps': 2000000000,
    'epsilon_model': 'exponential',
    'train_fc':'last2' # other options are last4,last3
}


def save_network_path(cfg, algorithm_cfg):
    # Save the network to the directory network_path
    weights_type = 'Imagenet'
    if algorithm_cfg.custom_load == True:
        algorithm_cfg['network_path'] = 'models/trained/' + cfg.env_type + '/' + cfg.env_name + '/' + 'CustomLoad/' + algorithm_cfg.train_type + '/'
    else:
        algorithm_cfg['network_path'] = 'models/trained/' + cfg.env_type + '/' + cfg.env_name + '/' + weights_type + '/' + algorithm_cfg.train_type + '/'

    if not os.path.exists(algorithm_cfg.network_path):
        os.makedirs(algorithm_cfg['network_path'])

    return cfg, algorithm_cfg


def get_new_position():
    x = int(input("x_val: "))
    y = int(input("y_val: "))
    z = int(input("z_val: "))

    position ={
        'x':x,'y':y,'z':z
    }
    return position

def get_new_orientation():
    Rx = int(input("Rx_val: "))
    Ry = int(input("Ry_val: "))
    Rz = int(input("Rz_val: "))

    orientation ={
        'x':Rx,'y':Ry,'z':Rz
    }
    return orientation

def move_drone(drone):
    position = get_new_position()
    orientation = get_new_orientation()
    new_pose = drone.get_new_pose(position,orientation)
    success = drone.set_to_new_pose(new_pose)
    return success

def move_drone_by_position(drone,current_state):
    current_position = current_state.kinematics_estimated.position
    current_yaw = airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)[2] * 57.2958  # Convert radians to degrees
    print(current_position)
    print(current_yaw)

    d_position= get_new_position().values()
    # print(d_position)
    dx,dy,dz = d_position

    new_x = current_position.x_val + dx
    new_y = current_position.y_val + dy
    new_z = current_position.z_val + dz

    success = drone.move_by_position(new_x,new_y,new_z)
    print(f"Moving drone2: {success}")







def choose_temp_action_selection2(drone):
    automate =True
    print("1: create drone client")
    print("2: enable API control")
    print("3: arm disarm")
    print("4: check api control is enabled or not")
    print("5: move drone by position")
    print("6: take off")
    print("7: reset")
    print("8: land drone")
    print("9: Current State")
    print("10: Barometer Sensor Data")
    print("11: IMU Sensor Data")
    print("12: GPS Sensor Data")
    print("13: Magnetometer Sensor Data")
    print("14: Lidar Sensor Data")
    print("15: GetImages")
    print("16: move drone by velocity")
    print("99: to terminate")
    print("its necessary to set 1,2,3 in first three tries to execute other commands")
    while automate:
        
        option = int(input("Enter your choice: "))
        # print(type(option))
        if option == 99:
            automate=False
        elif option == 1:
             drone.create_client()
        elif option == 2:
             drone.set_api_control()
        elif option == 3:
             drone.set_arm_disarm(),
        elif option == 4:
             print(drone.get_api_control_status()),
        elif option == 5:
            #  print(f"Moving drone: {move_drone(drone)}")
             current_state = drone.get_client_state()
             move_drone_by_position(drone,current_state)
        elif option == 6:
            drone.set_take_off()
        elif option == 7:
            drone.set_reset_drone()
        elif option == 8:
            drone.land_drone()
        elif option == 9:
            current_state = drone.get_client_state()
        elif option ==10:
            drone.get_barometer_data(vehicle_name = config['name_agent'])
        elif option ==11:
            drone.get_imu_data(vehicle_name = config['name_agent'])
        elif option ==12:
            drone.get_gps_data(vehicle_name = config['name_agent'])
        elif option ==13:
            drone.get_magnetometer_data(vehicle_name=config['name_agent'])
        elif option ==14:
            drone.get_lidar_data(vehicle_name=config['name_agent'])
        elif option ==15:
            drone.get_drone_images()
        elif option == 16:
            vx = float(input("Enter velocity in x direction (m/s): "))
            vy = float(input("Enter velocity in y direction (m/s): "))
            vz = float(input("Enter velocity in z direction (m/s): "))
            duration = float(input("Enter duration (seconds): "))
            success = drone.move_by_velocity(vx, vy, vz, duration)
            print(f"Move by velocity successful: {success}")
            # move_drone2(drone,current_state)
        
        
        # time.sleep(1)
        # switch_case(drone,option)



class DroneUI:
    def __init__(self, master, drone):
        self.master = master
        self.master.title("Drone Controller")

        # Show the list of options
        self.options_label = tk.Label(master, text="Available Options:")
        self.options_label.pack()

        self.options_text = tk.Text(master, height=15, width=50)
        self.options_text.pack()
        self.options_text.insert(tk.END, 
            "1: create drone client\n"
            "2: enable API control\n"
            "3: arm disarm\n"
            "4: check API control is enabled or not\n"
            "5: move drone by position\n"
            "6: take off\n"
            "7: reset\n"
            "8: land drone\n"
            "9: Current State\n"
            "10: Barometer Sensor Data\n"
            "11: IMU Sensor Data\n"
            "12: GPS Sensor Data\n"
            "13: Magnetometer Sensor Data\n"
            "14: Lidar Sensor Data\n"
            "15: GetImages\n"
            "16: move drone by velocity \n"
            "99: to terminate\n"
        )
        self.options_text.config(state=tk.DISABLED) 
        
        self.option_label = tk.Label(master, text="Enter option number:")
        self.option_label.pack()
        
        self.option_entry = tk.Entry(master)
        self.option_entry.pack()

        
        self.option_entry.bind('<Return>', self.run_command)
        
        self.run_button = tk.Button(master, text="Run Command", command=self.run_command)
        self.run_button.pack()

        self.clear_button = tk.Button(master, text="Clear Output", command=self.clear_output)
        self.clear_button.pack()
        
        self.output_label = tk.Label(master, text="Output:")
        self.output_label.pack()
        
        self.output_text = scrolledtext.ScrolledText(master, width=50, height=20)
        self.output_text.pack()
        
        self.drone = drone

    
    
    def run_command(self,event=None):
        option = self.option_entry.get()
        
        try:
            option = int(option)
            if option == 1:
                self.drone.create_client()
                self.update_output("Client created.")
            elif option == 2:
                self.drone.set_api_control()
                self.update_output("API control enabled.")
            elif option == 3:
                self.drone.set_arm_disarm()
                self.update_output("Drone armed/disarmed.")
            elif option == 4:
                flag = self.drone.get_api_control_status()
                self.update_output(f"API control status: {flag}")
            elif option == 5:
                new_position = self.get_position_input_dialog()
                if new_position:
                    success = self.drone.move_drone_by_position2(new_position)
                    # new_state = self.drone.get_client_state()
                    self.update_output(f"Drone moved successfully : {success}")
                else:
                    self.update_output("Invalid position input.")
                # self.drone.move_drone_by_position()
                # new_state = self.drone.get_client_state()
                # self.update_output(f"Drone moved to : {new_state}")
            elif option == 6:
                self.drone.set_take_off()
                self.update_output("Drone took off.")
            elif option == 7:
                self.drone.set_reset_drone()
                self.update_output("Drone reset.")
            elif option == 8:
                self.drone.land_drone()
                self.update_output("Drone landed.")
            elif option == 9:
                current_state = self.drone.get_client_state()
                self.update_output(f"Current state: {current_state}")
            elif option == 10:
                barometer_data = self.drone.get_barometer_data(vehicle_name=config['name_agent'])
                self.update_output(f"Barometer data: {barometer_data}")
            elif option == 11:
                imu_data = self.drone.get_imu_data(vehicle_name=config['name_agent'])
                self.update_output(f"IMU data: {imu_data}")
            elif option == 12:
                gps_data = self.drone.get_gps_data(vehicle_name=config['name_agent'])
                self.update_output(f"GPS data: {gps_data}")
            elif option == 13:
                magnetometer_data = self.drone.get_magnetometer_data(vehicle_name=config['name_agent'])
                self.update_output(f"Magnetometer data: {magnetometer_data}")
            elif option == 14:
                tmp_dir = self.drone.get_lidar_data(vehicle_name=config['name_agent'])
                self.update_output(f"Captured lidar point cloud and stored in {tmp_dir}.")
            elif option == 15:
                tmp_dir = self.drone.get_drone_images()
                self.update_output(f"Captured images and stored in {tmp_dir}.")
            elif option == 16:
                vx,vy,vz,duration = self.get_velocity_input_dialog()
                
                if vx is not None and vy is not None and vz is not None and duration is not None:
                    success = self.drone.move_by_velocity(vx, vy, vz, duration)
                    self.update_output(f"Move by velocity successful: {success}")
                else:
                    self.update_output("Invalid input for velocities or duration.")
            elif option == 99:
                self.master.quit()
            else:
                self.update_output("Invalid option.")
        except ValueError:
            self.update_output("Please enter a valid option number.")

        self.option_entry.delete(0, tk.END)
    
    def update_output(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def get_position_input_dialog(self):
        # Open a dialog to get custom x, y, z values
        x_val = simpledialog.askinteger("Input", "Enter x value:")
        y_val = simpledialog.askinteger("Input", "Enter y value:")
        z_val = simpledialog.askinteger("Input", "Enter z value:")

        if x_val is not None and y_val is not None and z_val is not None:
            pos ={'x': x_val, 'y': y_val, 'z': z_val}
            print(f"entered_values: {pos}")
            return pos
        else:
            return None
        
    def get_velocity_input_dialog(self):
        vx = simpledialog.askfloat("Input", "Enter velocity in x direction (m/s):")
        vy = simpledialog.askfloat("Input", "Enter velocity in y direction (m/s):")
        vz = simpledialog.askfloat("Input", "Enter velocity in z direction (m/s):")
        duration = simpledialog.askfloat("Input", "Enter duration (seconds):")

        if vx is not None and vy is not None and vz is not None and duration is not None:
            print(vx,vy,vz,duration)
            return vx,vy,vz,duration
        else:
            return None
    

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)


def testing_drone():
    # client, old_posit, initZ = connect_drone(ip_address=config['ip_address'], phase=config['mode'])
    name_agent =config['name_agent']
    root = tk.Tk()
    a = DroneAgent(deep_q_learning_config, {}, name='DQN', vehicle_name=name_agent)
    # action = choose_temp_action_selection2(a)

    app = DroneUI(root, a)
    root.mainloop()


    # action_word = translate_action(action, deep_q_learning_config['num_actions'])

    # time.sleep(5)

    # client.simPrintLogMessage("Status:", "Connected", 0)
    # time.sleep(5)

    # a.take_action(action, deep_q_learning_config['num_actions'], Mode='static')
    print("printing agent state")
    print(a.get_client_state())

    sys.exit()


def plot_training_rewards(rewards, save_path="training_rewards.png"):
    """
    Plot and save the training rewards over episodes.
    Includes a moving average for smoother visualization.

    Args:
        rewards (list): List of rewards collected over episodes.
        save_path (str): Path to save the rewards plot.
    """
    # Compute a moving average for smoother visualization
    moving_avg = np.convolve(rewards, np.ones(10) / 10, mode="valid")  # Moving average over 10 episodes

    # Plot rewards and moving average
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Rewards", alpha=0.7)
    plt.plot(range(len(moving_avg)), moving_avg, label="Moving Avg (10 episodes)", color="orange", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid()

    # Save the plot
    plt.savefig(save_path, format="png", dpi=300)
    print(f"Reward plot saved to: {save_path}")

    # Show the plot
    plt.show()


def testing_DQN():
    name_agent =config['name_agent']
    client, old_posit, initZ = connect_drone(ip_address=config['ip_address'])
    drone_agent = DroneAgent(deep_q_learning_config, client, name='DQN', vehicle_name=name_agent)

    drone_agent.create_client()
    drone_agent.set_api_control()
    drone_agent.set_arm_disarm()
    drone_agent.get_api_control_status()

    # Training loop
    for episode in range(500):  # Number of episodes
        state = drone_agent.get_sensor_data()
        total_reward = 0

        for t in range(200):  # Max steps per episode
            action = drone_agent.select_action(state)
            drone_agent.move_by_velocity(action)

            next_state = drone_agent.get_sensor_data()
            if episode == 2:
                print("State vector shape:", next_state.shape)
            collision_info = client.simGetCollisionInfo(vehicle_name=config['name_agent'])
            reward = drone_agent.get_reward(collision_info)
            done = collision_info.has_collided

            drone_agent.store_experience(state, action, reward, next_state, done)
            drone_agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        # Decay epsilon
        drone_agent.epsilon = max(drone_agent.min_epsilon, drone_agent.epsilon * drone_agent.epsilon_decay)

        # Update target network periodically
        if episode % 10 == 0:
            drone_agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}")

        # Assuming 'rewards' contains the total rewards for all episodes
    # plt.plot(rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("Training Rewards")

    # # Save the plot as an image file
    # plt.savefig("training_rewards.png", format="png", dpi=300)  # Save as PNG with high resolution

    # # Display the plot
    # plt.show()

    # Cleanup
    env_process.terminate()
    
# Add the `InitializeNetworkDeepQLearning` Class Here (from the rewritten PyTorch-based code)


# Ensure this class is available before integrating with the DroneAgent class.

class InitializeNetworkDeepQLearning(nn.Module):
    def __init__(self, cfg, name, vehicle_name):
        """
        Initialize the Deep Q-Learning Network in PyTorch.
        Args:
            cfg (dict): Configuration dictionary containing hyperparameters.
            name (str): Name of the network.
            vehicle_name (str): Name of the vehicle.
        """
        super(InitializeNetworkDeepQLearning, self).__init__()
        self.vehicle_name = vehicle_name
        self.name = name
        self.cfg = cfg

        # Model parameters
        self.input_size = cfg["input_size"]  # Input dimensions (e.g., state size)
        self.num_actions = cfg["num_actions"]  # Number of actions
        self.learning_rate = cfg.get("learning_rate", 1e-3)  # Learning rate

        # Define the Q-network
        self.model = self.create_network()

        # Loss function and optimizer
        self.loss_fn = nn.SmoothL1Loss()  # Equivalent to Huber loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))

        # Replay buffer (used for storing experiences during training)
        self.experience_replay = []

        # Training-specific parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Network initialized for {self.vehicle_name} with model: {self.model}")

    def create_network(self):
        """
        Create the neural network architecture (similar to C3F2 in original code).
        Replace this with the desired architecture as needed.
        """
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # Conv Layer 1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Conv Layer 2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Conv Layer 3
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # Fully Connected Layer 1 (adjust size based on input resolution)
            nn.ReLU(),
            nn.Linear(512, self.num_actions)  # Fully Connected Layer 2 (output layer)
        )

    def forward(self, state):
        """
        Forward pass through the Q-network.
        Args:
            state (torch.Tensor): Input state tensor (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Q-values for each action.
        """
        return self.model(state)

    def train_batch(self, states, targets, actions):
        """
        Train the Q-network on a batch of data.
        Args:
            states (torch.Tensor): Batch of states (input to the network).
            targets (torch.Tensor): Target Q-values (calculated from Bellman equation).
            actions (torch.Tensor): Actions taken for each state in the batch.
        Returns:
            float: Loss value for the current batch.
        """
        self.model.train()

        # Forward pass
        q_values = self(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute loss
        loss = self.loss_fn(q_values, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict_q_values(self, state):
        """
        Predict Q-values for a given state.
        Args:
            state (torch.Tensor): State tensor (single or batch).
        Returns:
            torch.Tensor: Predicted Q-values for all actions.
        """
        self.model.eval()
        with torch.no_grad():
            return self(state)

    def select_action(self, state, epsilon):
        """
        Epsilon-greedy action selection.
        Args:
            state (torch.Tensor): State tensor (single input).
            epsilon (float): Exploration probability.
        Returns:
            int: Selected action index.
        """
        if np.random.rand() < epsilon:
            # Explore: Choose a random action
            return np.random.randint(self.num_actions)
        else:
            # Exploit: Choose the action with the highest Q-value
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch dimension
            q_values = self.predict_q_values(state)
            return torch.argmax(q_values).item()

    def save_model(self, save_path):
        """
        Save the model to the specified path.
        Args:
            save_path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        Load the model from the specified path.
        Args:
            load_path (str): Path to load the model from.
        """
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        """
        Add a new experience to the replay buffer.
        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state after the action.
            done (bool): Whether the episode ended.
        """
        if len(self.experience_replay) >= self.cfg.get("buffer_len", 10000):
            self.experience_replay.pop(0)  # Remove the oldest experience if the buffer is full
        self.experience_replay.append((state, action, reward, next_state, done))

    def sample_from_replay_buffer(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer.
        Args:
            batch_size (int): Number of experiences to sample.
        Returns:
            list: Batch of experiences (state, action, reward, next_state, done).
        """
        return random.sample(self.experience_replay, batch_size)

    def train_from_replay_buffer(self, batch_size):
        """
        Train the Q-network using experiences from the replay buffer.
        Args:
            batch_size (int): Number of experiences to sample for training.
        """
        if len(self.experience_replay) < batch_size:
            return  # Not enough data to train

        # Sample a batch
        batch = self.sample_from_replay_buffer(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute targets using the Bellman equation
        next_q_values = self.predict_q_values(next_states).max(dim=1)[0]
        targets = rewards + (1 - dones) * self.cfg.get("gamma", 0.99) * next_q_values

        # Train the network
        loss = self.train_batch(states, targets, actions)
        return loss



# Define a mapping of action indices to velocities (example values)
velocity_mapping = [
    [1, 0, 0],  # Action 0: Move in x direction
    [0, 1, 0],  # Action 1: Move in y direction
    [0, 0, 1],  # Action 2: Move up in z direction
    [-1, 0, 0], # Action 3: Move in -x direction
    [0, -1, 0], # Action 4: Move in -y direction
    [0, 0, -1], # Action 5: Move down in z direction
]


### **Integrating with `testing_DQN`**
def generate_velocity(action, num_actions):
    # Divide the action space into velocity directions (example logic)
    step_size = 2 / num_actions  # Step size based on the number of actions
    vx = (action % 5 - 2) * step_size  # Cycles through -2 to 2 for x
    vy = ((action // 5) % 5 - 2) * step_size  # Cycles through -2 to 2 for y
    vz = 0  # Fixed z velocity (you can modify this as needed)
    return [vx, vy, vz]

def testing_DQN2():
    name_agent = config['name_agent']
    client, old_posit, initZ = connect_drone(ip_address=config['ip_address'])
    drone_agent = DroneAgent(deep_q_learning_config, client, name='DQN', vehicle_name=name_agent)

    drone_agent.create_client()
    drone_agent.set_api_control()
    drone_agent.set_arm_disarm()
    drone_agent.get_api_control_status()

    initial_position = client.simGetVehiclePose(vehicle_name=config['name_agent']).position


    rewards = []  # To store total rewards per episode

    # Training loop
    for episode in range(500):  # Number of episodes
        state = drone_agent.get_sensor_data()
        # state_tensor = torch.FloatTensor(state).unsqueeze(0).view(1, 3, 64, 64)
        total_reward = 0

        for t in range(200):  # Max steps per episode
            action = drone_agent.select_action(state)
            print(action)
            # velocity = velocity_mapping[action]  # Map action index to velocity
            velocity = generate_velocity(action, drone_agent.cfg['num_actions'])
            drone_agent.move_by_velocity(velocity[0], velocity[1], velocity[2], 5)  # Move the drone

            # velocity = drone_agent.cfg['num_actions'][action]  # Map action to velocity
            # drone_agent.move_by_velocity(velocity[0], velocity[1], velocity[2], 1)

            collision_info = client.simGetCollisionInfo(vehicle_name=config['name_agent'])
            current_position = client.simGetVehiclePose(vehicle_name=config['name_agent']).position
            

            reward = drone_agent.get_reward(collision_info,initial_position)

            # Check for collision or boundary violation
            if collision_info.has_collided or abs(current_position.x_val) > 50 or abs(current_position.y_val) > 50 or abs(current_position.z_val) > 30:
                print("Collision or boundary violation detected. Resetting...")
                drone_agent.move_to_initial_position(initial_position)
                # reward = drone_agent.get_reward(collision_info,initial_position)
                # drone_agent.create_client()
                # drone_agent.set_api_control()
                # drone_agent.set_arm_disarm()
                # drone_agent.get_api_control_status()
                done = True  # Episode terminates if a collision or boundary violation occurs
            else:
                done = False


            
            # done = collision_info.has_collided

            next_state = drone_agent.get_sensor_data()
            # reward = drone_agent.get_reward(collision_info,initial_position)

            drone_agent.store_experience(state, action, reward, next_state, done)
            drone_agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        # Decay epsilon
        drone_agent.epsilon = max(drone_agent.epsilon_min, drone_agent.epsilon * drone_agent.epsilon_decay)

        # Update target network periodically
        if episode % 10 == 0:
            drone_agent.update_target_network()

        rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")

    # Plot training rewards
    plot_training_rewards(rewards, save_path="training_rewards.png")



if __name__ == '__main__':
    can_proceed = generate_json(config)

    env_name ="MyProject2_4point_27"
    config['env_name'] = env_name
    env_process, env_folder = start_environment(env_name)
    print(config)

    # main()


    # testing droneAgent
    # testing_drone()

    # DQN
    testing_DQN2()

    # DeepQLearning(config,env_process, env_folder)


    

    # print(nvidia_smi.nvmlInit())