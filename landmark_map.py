import numpy as np
import matplotlib.pyplot as plt

class LandmarkMap:
    def __init__(self,n_landmark):
        self.n_landmark = n_landmark
        self.landmarks = np.zeros([3,self.n_landmark])
        self.eye_3m = np.eye(3*self.n_landmark)
        self.cov = self.eye_3m # 3*N_landmark x 3*N_landmark

    def get_landmark(self,idxes):
        return self.landmarks[:,idxes]

    def plot_map(self):
        plt.scatter(self.landmarks[0], self.landmarks[1], 1)

    def update_cov(self,K,H):
        self.cov = (self.eye_3m - K @ H) @ self.cov
    
    def update_mean(self,stero_cam,landmark_idx,kalman_gain):
        pass
    
