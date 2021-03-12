# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from tqdm import tqdm_notebook
import utils
from transform import Transform
from sensors import IMU, StereoCamera
from mapping import LandmarkMap, SLAM
from kalman_filter import KalmanFilter
# %%
filename = "./data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = utils.load_data(filename, load_features = True)

# downsample features
features = features[:,::10,:]
#%%
stero_cam = StereoCamera(K,b,features,noise=10)
tf = Transform()
imu = IMU(t, linear_velocity, angular_velocity, noise=1e-5)
kf = KalmanFilter()

# %% run loop for update-
data_length = t.shape[1]
slam = SLAM(n_landmark=features.shape[1], imu=imu, cov_init=1)

for t_idx in tqdm_notebook(range(data_length)):
    # predict pose mean and cov
    robot_pose = slam.predict_pose_mean(t_idx)
    slam.predict_cov_combined(t_idx)

    # separate observed landmarks. old: landmk seen before, new: unseen landmk
    landmark_idx = stero_cam.get_landmark_seen(t_idx) # all landmarks in current frame
    old_landmk_idx, new_landmk_idx = slam.landmark_map.get_old_new_landmarks(landmark_idx)

    # initialize new landmarks
    if len(new_landmk_idx) > 0:
        pixel_new = stero_cam.get_landmark_freatures(new_landmk_idx, t_idx)
        xyz_optical = stero_cam.pixel_to_xyz(pixel_new, max_depth=50)
        xyz_world = tf.optical_to_world(robot_pose, xyz_optical) # in homogenous
        slam.landmark_map.init_landmarks(new_landmk_idx,xyz_world[:3])
        
    # update old landmarks and poses using EKF
    if len(old_landmk_idx) > 0:
        # get landmark seen in current frame
        pixels_obs = stero_cam.get_landmark_freatures(old_landmk_idx, t_idx) #pixel features
        landmks_xyz = slam.landmark_map.get_landmarks(old_landmk_idx) # world coordinates

        # set patches for landmark partial covariance and mean update
        slam.set_update_patch(old_landmk_idx)

        # calculate jacobian and kalman gain
        H_pose = kf.calculate_pose_jacobian(stero_cam,tf,robot_pose,landmks_xyz) # 4N x 6
        H_lmk = kf.calculate_observation_jacobian(stero_cam,tf,robot_pose,landmks_xyz) # 4N x 3N
        H_combine = np.hstack((H_pose, H_lmk))
        cov_patch = slam.get_cov_combined_patch() # part of cov matching landmk seen
        K_combine = kf.calculate_kalman_gain(cov_patch, H_combine, stero_cam.cov)

        # update mean and covariance of pose + landmark state, EKF
        innovation = utils.calcualte_innovation(stero_cam,tf,robot_pose,landmks_xyz,pixels_obs)
        slam.update_pose_landmark_mean(K_combine,innovation,t_idx)
        slam.update_pose_landmark_cov(cov_patch,K_combine,H_combine)
    
    # if no EKF then don't update pose, only use predict
    else:
        slam.pose_tracker.skip_update(t_idx)

# %%
utils.visualize_trajectory_2d(slam.pose_tracker.poses_ekf,
                              landmarks=slam.landmark_map.landmarks,
                              show_ori=True)

# %%
