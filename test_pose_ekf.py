# %%
from mapping import PoseTracker
from kalman_filter import KalmanFilter
from mapping import LandmarkMap
from sensors import IMU, StereoCamera
from transform import Transform
import numpy as np
import matplotlib.pyplot as plt
import utils
import importlib
from scipy.linalg import expm
from tqdm.notebook import tqdm_notebook

import transform
import sensors
import mapping
import kalman_filter
importlib.reload(transform)
importlib.reload(sensors)
importlib.reload(mapping)
importlib.reload(kalman_filter)

# %%
filename = "./data/10.npz"
t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = utils.load_data(
    filename, load_features=True)

# downsample features
features = features[:, ::20, :]

# %% initialize
imu = IMU(t, linear_velocity, angular_velocity)
tf = Transform()
stero_cam = StereoCamera(K, b, features)
myMap = LandmarkMap(stero_cam.n_features)
pose_tracker = PoseTracker(imu)

# %% predict all poses using pose kinematics
for t_idx in range(t.shape[1]):
    pose_tracker.predict_pose(t_idx)

utils.visualize_trajectory_2d(pose_tracker.poses_pred)
# %% test dead reconking map
data_length = t.shape[1]
pose_all = pose_tracker.poses_pred

for idx in range(data_length):
    robot_pose = pose_all[:, :, idx]
    landmark_idx = stero_cam.get_landmark_seen(idx)
    pixel_feature = stero_cam.get_landmark_freatures(landmark_idx, idx)
    xyz_optical = stero_cam.pixel_to_xyz(pixel_feature, max_depth=50)
    xyz_world = tf.optical_to_world(robot_pose, xyz_optical)
    myMap.landmarks[:, landmark_idx] = xyz_world[:3, :]

myMap.plot_map()
plt.show()

# %% calculate pose based on EKF
imu = IMU(t, linear_velocity, angular_velocity, noise=0.0005)
pose_tracker = PoseTracker(imu)
data_length = t.shape[1]
kf = KalmanFilter()

for t_idx in tqdm_notebook(range(data_length)):
    # predict mean and cov
    pose_tracker.predict_pose(t_idx)
    pose_tracker.predict_covariance(t_idx)

    # update
    landmark_idx = stero_cam.get_landmark_seen(t_idx)
    if len(landmark_idx) > 0:
        # observed landmark pixel features
        pixel_obs = stero_cam.get_landmark_freatures(landmark_idx, t_idx)
        landmks_xyz = myMap.get_landmarks(landmark_idx) #landmark world coordinates

        # calculate jacobian, Kalman gain
        pose_pred = pose_tracker.poses_pred[:, :, t_idx]
        cov_pred = pose_tracker.cov_all[:, :, t_idx]
        jacobian = kf.calculate_pose_jacobian(stero_cam, tf, pose_pred, landmks_xyz)
        K_gain = kf.calculate_kalman_gain(cov_pred, jacobian, stero_cam.cov)

        # update mean and cov using EKF
        pose_tracker.update_pose(stero_cam, tf, landmks_xyz, pixel_obs, K_gain, t_idx)
        pose_tracker.update_covariance(K_gain,jacobian,t_idx)
    else:
        pose_tracker.skip_update(t_idx)
# %%
utils.visualize_trajectory_2d(pose_tracker.get_final_trajectory(ekf_pose=False))
# %%
