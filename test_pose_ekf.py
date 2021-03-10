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
from mapping import LandmarkMap
from kalman_filter import KalmanFilter
from mapping import PoseTracker
# %%
filename = "./data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)

# downsample features
features = features[:,::10,:]

# %% initialize
imu = IMU(t, linear_velocity, angular_velocity)
tf = Transform()
stero_cam = StereoCamera(K,b,features)
myMap = LandmarkMap(stero_cam.n_features)
pose_tracker = PoseTracker(imu)

# %% predict all poses using pose kinematics
for t_idx in range(t.shape[1]):
    pose_tracker.predict_pose(t_idx)

visualize_trajectory_2d(pose_tracker.poses_pred)
# %% test dead reconking map
data_length = t.shape[1]
pose_all = pose_tracker.poses_pred

for idx in range(data_length):
    robot_pose = pose_all[:,:,idx]
    landmark_idx = stero_cam.get_landmark_seen(idx)
    pixel_feature = stero_cam.get_landmark_freatures(landmark_idx, idx)
    xyz_optical = stero_cam.pixel_to_xyz(pixel_feature, max_depth=50)
    xyz_world = tf.optical_to_world(robot_pose, xyz_optical)
    myMap.landmarks[:,landmark_idx] = xyz_world[:3,:]

myMap.plot_map()
plt.show()

# %% 
kf = KalmanFilter()
# %%
kf._get_pose_jacobian_block(stero_cam,tf,myMap.landmarks[:,0],pose_tracker.poses_pred[:,:,0])
# %%
