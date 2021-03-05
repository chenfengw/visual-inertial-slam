# %%
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import importlib
from scipy.linalg import expm

import transform
import sensors
importlib.reload(transform)
importlib.reload(sensors)

from transform import Transform
from sensors import IMU
from sensors import StereoCamera
# %%
filename = "./data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)

# %% plot dead reconking
imu = IMU(t, linear_velocity, angular_velocity)
tao = np.diff(imu.get_time()).mean()
# %%
pose_all = np.zeros([4,4,imu.get_length()])
pose_all[:,:,0] = Transform.calcualte_pose(np.eye(3),np.zeros(3))

for idx in range(imu.get_length()-1):
    v = imu.linear_velocity[:,idx]
    omega = imu.angular_velocity[:,idx]
    twist = Transform.calculate_twist(np.concatenate([v,omega]))
    pose_all[:,:,idx+1] = pose_all[:,:,idx] @ expm(tao * twist)
# %%
visualize_trajectory_2d(pose_all,show_ori=True)
# %% test stero camera
stero_cam = StereoCamera(K,b)

#%% test
x1 = features[:,:,0]
valid = x1[0] != -1
x_valid = x1[:,valid]
# %%
for idx in range(1):
    pixel_cooridnates = features[:,:,idx]
    valid = pixel_cooridnates[0] != -1
    pixel_valid = pixel_cooridnates[:,valid]
    
    d = pixel_valid[0] - pixel_valid[2]
    print(sum(d<0))
# %%
xyz = stero_cam.pixel_to_xyz(pixel_valid)
# %%
tf = Transform(imu_T_cam)
X = tf.optical_to_world(pose_all[:,:,0],xyz)
# %%
