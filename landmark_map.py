import numpy as np
import matplotlib.pyplot as plt

class LandmarkMap:
    def __init__(self,n_landmark):
        self.n_landmark = n_landmark
        self.landmarks = np.zeros([3,self.n_landmark])
        self._eye_3m = np.eye(3*self.n_landmark)
        self.cov = self._eye_3m * 0.01 # 3*N_landmark x 3*N_landmark
        self._landmarks_seen = set()
    
    def get_landmarks(self,idxes):
        return self.landmarks[:,idxes]

    def init_landmarks(self,idxes,xyz_coordinate):
        # only initialize new landmarks
        assert xyz_coordinate.shape[0] == 3
        self.landmarks[:,idxes] = xyz_coordinate

    def plot_map(self):
        plt.scatter(self.landmarks[0], self.landmarks[1], 1)

    def update_cov(self,K,H):
        self.cov = (self._eye_3m - K @ H) @ self.cov
    
    def update_landmarks(self, stero_cam, tf, world_T_imu, landmk_xyz, landmk_obs, K_gain):
        # get observed pixels
        landmk_pred = stero_cam.xyz_to_pixel(tf, world_T_imu, landmk_xyz)
        innovation = landmk_obs - landmk_pred # 4 x N_t

        # update mean using EKF
        mu_new = self.landmarks.ravel(order="F") + K_gain @ innovation.ravel(order="F")
        self.landmarks = mu_new.reshape(self.landmarks.shape,order="F")
        
    def get_old_new_landmarks(self, landmark_current_frame):
        # get landmarks that has seen before, and new landmarks
        landmarks_old = self._landmarks_seen.intersection(landmark_current_frame)
        landmarks_new = set(landmark_current_frame) - landmarks_old
        
        # update landmark seen
        self._landmarks_seen.update(landmark_current_frame)
        return list(landmarks_old), list(landmarks_new)