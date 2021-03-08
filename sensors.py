import numpy as np
import pandas as pd
import math

class IMU():
    def __init__(self, t, linear_v, angular_v):
        self.t = t.flatten()
        self.linear_velocity = linear_v
        self.angular_velocity = angular_v
        self.delta_t = np.diff(self.t).mean() # time interval between consecutive updates

    def get_time(self):
        return self.t
    
    def get_length(self):
        return len(self.t)


class StereoCamera():
    def __init__(self, K, b, features, noise=5):
        self.K = K # intrinsic matrix, 3x3
        self.b = b # baseline, distance between two camera
        self.fsu = self.K[0,0]
        self.fsv = self.K[1,1]
        self.cu = self.K[0,2]
        self.cv = self.K[1,2]
        self.fsub = self.fsu * self.b
        self.M = self.get_calibration_matrix()
        self.features = features  # 4 x n_features x n_frame
        self.n_features = features.shape[1]
        self.features_idx = np.arange(self.n_features)
        self.cov = np.eye(4) * noise # noise covariance matrix

    def get_calibration_matrix(self):
        M = np.zeros([4,4])
        temp = self.K[:2,:]
        M[:2,:3] = temp
        M[-2:,:3] = temp
        M[2,3] = -self.fsub
        return M

    def get_landmark_seen(self, frame_idx):
        """Get landmark index that camera sees in a frame. 

        Args:
            frame_idx (int): index of the frame

        Returns:
            tuple: index of all features in given frame, pixel coordinates
        """
        pixels = self.features[:,:,frame_idx]
        valid = pixels[0] != -1 # valid features, invalid features is -1
        feature_idx = self.features_idx[valid]

        return feature_idx

    def get_landmark_freatures(self, landmark_idx, frame_idx):
        return self.features[:,landmark_idx,frame_idx]

    def pixel_to_xyz(self, pixels, max_depth=25):
        """Given pixel coordinates find out xyz in camera frame

        Args:
            pixels (np array): 4 x N_features
            max_depth (int): set max pixel depth in meter
        Returns:
            np array: xyz coordinates of pixels in homogenous coordinates, 
            3 (x,y,z) x N_features
        """
        assert pixels.shape[0] == 4
        d = np.abs(pixels[0] - pixels[2]) # disparity U_L - U_R
        z = self.fsub / d
        z[z > max_depth] = max_depth
        
        # calcualte xy
        u_L = pixels[0] # take first row
        v_L = pixels[1] # take 2nd row
        x = (u_L - self.cu) / self.fsu * z
        y = (v_L - self.cv) / self.fsv * z
        
        return np.vstack((x,y,z))

    def xyz_to_pixel(self, tf, world_T_imu, xyz_world):
        assert xyz_world.shape[0] == 3, "must have 3 rows"
        optical_xyz = tf.world_to_optical(world_T_imu,xyz_world) # in homogenous
        z = optical_xyz[2]
        optical_xyz /= z # divide xyz by 1
        return self.M @ (optical_xyz)

