import numpy as np
import matplotlib.pyplot as plt
from utils import get_patch_idx 
from transform import Transform
from scipy.linalg import expm

class BasicMap:
    def __init__(self,n_landmark):
        self.n_landmark = n_landmark
        self.landmarks = np.zeros([3,self.n_landmark])
        self._landmarks_seen = set()

    def get_landmarks(self,idxes):
        return self.landmarks[:,idxes]

    def init_landmarks(self,idxes,xyz_coordinate):
        # only initialize new landmarks
        assert xyz_coordinate.shape[0] == 3
        self.landmarks[:,idxes] = xyz_coordinate

    def get_old_new_landmarks(self, landmark_current_frame):
        # get landmarks that has seen before, and new landmarks
        landmarks_old = self._landmarks_seen.intersection(landmark_current_frame)
        landmarks_new = set(landmark_current_frame) - landmarks_old
        
        # update landmark seen
        self._landmarks_seen.update(landmark_current_frame)
        return list(landmarks_old), list(landmarks_new)

    def plot_map(self):
        plt.scatter(self.landmarks[0], self.landmarks[1], 1)

class LandmarkMap(BasicMap):
    def __init__(self,n_landmark,noise=0.01):
        super().__init__(n_landmark)
        self.cov = np.eye(3*self.n_landmark) * noise # 3*N_landmark x 3*N_landmark
        self._landmks_idx = None # landmark index seen in current frame
        self._cov_idx = None     # index of cov corresponding to current landmarks
        self.cov_patch = None    # patch of cov corresponding to current landmarks

    def update_cov(self,K,H):
        cov_patch = self.cov[np.ix_(self._cov_idx, self._cov_idx)] # 3N_t x 3N_t
        size = cov_patch.shape[0]
        cov_patch = (np.eye(size) - K @ H) @ cov_patch
        self.cov[np.ix_(self._cov_idx, self._cov_idx)] = cov_patch

    def update_landmarks(self, K_gain, innovation):
        # update mean using EKF
        mu_patch = self.get_landmarks(self._landmks_idx).ravel(order="F") # 3N_t x 1
        mu_partial = mu_patch + K_gain @ innovation.ravel(order="F")
        self.landmarks[:,self._landmks_idx] = mu_partial.reshape((3,-1),order="F")
    
    def set_current_patch(self,landmark_idxs):
        self._landmks_idx = landmark_idxs
        self._cov_idx = get_patch_idx(self._landmks_idx)

class PoseTracker:
    def __init__(self,imu):
        self.imu = imu
        self.n_poses = self.imu.get_length()
        self.poses_pred = np.zeros([4,4,self.n_poses])
        self.poses_ekf = np.zeros_like(self.poses_pred)
        self.cov_all = np.zeros([6,6,self.n_poses])
        self._eye6 = np.eye(6)

    def predict_pose(self,t_idx):
        if t_idx == 0:
            self.poses_pred[:,:,t_idx] = Transform.calcualte_pose(np.eye(3),
                                                                  np.zeros(3))
        else:
            u = self.imu.get_linear_angular_velocity(t_idx)
            twist = Transform.calculate_twist(u)
            self.poses_pred[:,:,t_idx] = self.poses_pred[:,:,t_idx-1] @ expm(self.imu.delta_t * twist)
        return self.poses_ekf[:,:,t_idx]

    def predict_covariance(self,t_idx):
        if t_idx == 0:
            self.cov_all[:,:,t_idx] = self._eye6
        else:
            u = self.imu.get_linear_angular_velocity(t_idx)
            exp_arg = - self.imu.delta_t * Transform.adjoint_6d(u)
            cov_new = expm(exp_arg) @ self.cov_all[:,:,t_idx-1] @ expm(exp_arg).T + self.imu.W
            self.cov_all[:,:,t_idx] = cov_new

    def update_pose(self, K_gain, innovation, t_idx):
        exp_args = Transform.calculate_twist(K_gain @ innovation.ravel(order="F"))
        self.poses_ekf[:,:,t_idx] = self.poses_pred[:,:,t_idx] @ expm(exp_args)

    def update_covariance(self,K_gain,H,t_idx):
        self.cov_all[:,:,t_idx] = (self._eye6 - K_gain @ H) @ self.cov_all[:,:,t_idx]

    def skip_update(self,t_idx):
        self.poses_ekf[:,:,t_idx] = self.poses_pred[:,:,t_idx]

    def get_final_trajectory(self,ekf_pose=True):
        if ekf_pose:
            return self.poses_ekf
        else:
            return self.poses_pred

class SLAM():
    def __init__(self,n_landmark=None,imu=None):
        pose_tracker = PoseTracker(imu)
        landmark_map = LandmarkMap(n_landmark)
        self.cov_combine = np.zeros([6+3*n_landmark,6+3*n_landmark])

    def predict_pose_mean(self):
        pass

    def predict_pose_cov(self):
        pass

    def update_pose_landmark_mean(self):
        pass

    def update_pose_landmark_cov(self):
        pass