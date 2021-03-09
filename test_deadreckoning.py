# %%
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import importlib
from scipy.linalg import expm

import transform
import sensors
import mapping
import kalman_filter
importlib.reload(transform)
importlib.reload(sensors)
importlib.reload(mapping)
importlib.reload(kalman_filter)

from transform import Transform
from sensors import IMU, StereoCamera
from landmark_map import LandmarkMap
from kalman_filter import KalmanFilter
# %%
filename = "./data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)

# downsample features
features = features[:,::10,:]

# %% plot dead reconking
imu = IMU(t, linear_velocity, angular_velocity)
tf = Transform()
stero_cam = StereoCamera(K,b,features)
myMap = LandmarkMap(stero_cam.n_features)

# %%
pose_all = np.zeros([4,4,imu.get_length()])
pose_all[:,:,0] = Transform.calcualte_pose(np.eye(3),np.zeros(3))

for idx in range(imu.get_length()-1):
    v = imu.linear_velocity[:,idx]
    omega = imu.angular_velocity[:,idx]
    twist = Transform.calculate_twist(np.concatenate([v,omega]))
    pose_all[:,:,idx+1] = pose_all[:,:,idx] @ expm(imu.delta_t * twist)
# %%
visualize_trajectory_2d(pose_all,show_ori=True)

# %% test dead reconking map
data_length = t.shape[1]

for idx in range(data_length):
    robot_pose = pose_all[:,:,idx]
    landmark_idx = stero_cam.get_landmark_seen(idx)
    pixel_feature = stero_cam.get_landmark_freatures(landmark_idx, idx)
    xyz_optical = stero_cam.pixel_to_xyz(pixel_feature, max_depth=50)
    xyz_world = tf.optical_to_world(robot_pose, xyz_optical)
    myMap.landmarks[:,landmark_idx] = xyz_world[:3,:]

# %%
myMap.plot_map()
plt.show()
# %% test karmal KalmanFilter
kf = KalmanFilter()

# %% run loop for update-
# data_length = t.shape[1]
data_length = 2
myMap = LandmarkMap(stero_cam.n_features)

for idx in range(data_length):
    robot_pose = pose_all[:,:,idx]
    landmark_idx = stero_cam.get_landmark_seen(idx)
    pixel_feature = stero_cam.get_landmark_freatures(landmark_idx, idx)
    xyz_optical = stero_cam.pixel_to_xyz(pixel_feature, max_depth=50)
    xyz_world = tf.optical_to_world(robot_pose, xyz_optical)
    myMap.landmarks[:,landmark_idx] = xyz_world[:3,:]

    H = kf.calculate_observation_jacobian(stero_cam,tf,robot_pose,landmark_idx,myMap)
    kalman_gain = kf.calculate_kalman_gain(myMap.cov,H,stero_cam.cov)
    myMap.update_cov(kalman_gain,H)
