import numpy as np
import pandas as pd
import math

class IMU():
    def __init__(self, t, linear_v, angular_v):
        self.t = t.flatten()
        self.linear_velocity = linear_v
        self.angular_velocity = angular_v
    
    def get_time(self):
        return self.t
    
    def get_length(self):
        return len(self.t)


class StereoCamera():
    def __init__(self, K, b):
        self.K = K # intrinsic matrix, 3x3
        self.b = b # baseline, distance between two camera
        self.fsu = self.K[0,0]
        self.fsub = self.fsu * self.b
        self.M = self.get_calibration_matrix()

    def get_calibration_matrix(self):
        M = np.zeros([4,4])
        temp = self.K[:2,:]
        M[:2,:3] = temp
        M[-2:,:3] = temp
        M[2,3] = -self.fsub
        return M

    def pixel_to_xyz(self,pixels):
        """Given pixel coordinates find out xyz in camera frame

        Args:
            pixels (np array): 4 x N_features

        Returns:
            np array: xyz coordinates of pixels in homogenous coordinates, 
            4 (x,y,z,1) x N_features
        """
        assert pixels.shape[0] == 4
        xyz = np.linalg.pinv(self.M) @ pixels
        d = np.abs(pixels[0] - pixels[2]) # disparity U_L - U_R
        z = self.fsub / d

        # multiply z to and set last row to be 1
        xyz *= z
        xyz[2,:] = z
        xyz[-1,:] = 1
        return xyz
