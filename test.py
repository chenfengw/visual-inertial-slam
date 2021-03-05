# %%
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import importlib
from scipy.linalg import expm

import transform
import sensors
import landmark_map
importlib.reload(transform)
importlib.reload(sensors)
importlib.reload(landmark_map)

from transform import Transform
from sensors import IMU
from sensors import StereoCamera
from landmark_map import LandmarkMap
# %%
filename = "./data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)

# %% plot dead reconking
imu = IMU(t, linear_velocity, angular_velocity)
tf = Transform(imu_T_cam)
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
data_length = t.shape[-1]

for idx in range(data_length):
    robot_pose = pose_all[:,:,idx]
    landmark_idx, pixel_feature = stero_cam.get_frame(idx)
    xyz_optical = stero_cam.pixel_to_xyz(pixel_feature, max_depth=50)
    xyz_world = tf.optical_to_world(robot_pose, xyz_optical)
    myMap.landmarks[:,landmark_idx] = xyz_world

# %%
myMap.plot_map()
plt.show()
# %%
