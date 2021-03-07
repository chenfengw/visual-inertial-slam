import numpy as np
import utils as ut


class KalmanFilter:
    def __init__(self):
        pass

    @staticmethod
    def _get_observation_jacobian_block(stero, tf, landmark_xyz, world_T_imu):
        """Calculate each block of the observation jacobian matrix

        Args:
            stero (SteroCamera Object): stereo camera
            tf (Transform object): transform
            landmark_xyz (np array): xyz coordinates of landmark
            world_T_imu (np array): robot pose

        Returns:
            np array: 4x3 matrix
        """
        optical_T_world = tf.optical_T_imu @ tf.inverse(world_T_imu)
        q = optical_T_world  @ tf.make_homogenous(landmark_xyz)
        proj_deriv = ut.calcualte_projection_derivative(q)

        return stero.M @ proj_deriv @ optical_T_world @ tf.P.T

    @staticmethod
    def calculate_observation_jacobian(stero, 
                                       tf,
                                       world_T_imu,
                                       landmark_idxs, 
                                       landmark_map):
        """Calculate jacobian for observation model

        Args:
            stero (SteroCamera): camera object
            tf (Transform): transform object
            world_T_imu (np array): robot pose, 4x4
            landmark_idxs (np array): index of landmark seen in a frame. 1 x N_t
            landmark_map (LandmarkMap): map object, has M landmarks

        Returns:
            np array: jacobian matrix, 4Nt x 3M
        """
        n_features = len(landmark_idxs) #number of landmark seen
        n_landmark = landmark_map.n_landmark
        H = np.zeros((4*n_features, 3*n_landmark))

        for row_idx in range(n_features):
            landmark_idx = landmark_idxs[row_idx]
            landmark_xyz = landmark_map.get_landmark(landmark_idx)

            # update (row_idx, landmark_idx) block of H
            i = 4*row_idx
            j = 3*landmark_idx
            H[i:i+4, j:j+3] = KalmanFilter._get_observation_jacobian_block(stero, 
                                                                           tf, 
                                                                           landmark_xyz, 
                                                                           world_T_imu)
        return H

    @staticmethod
    def calculate_kalman_gain(cov,H,V):
        """Calculate Kalman gain

        Args:
            cov (np array): covariance of landmark xyz, 3mx3m matrix
            H (np array): jacobian, 4N_t x 3M. N_t is number of landmark seen
            in a frame, M is total number of landmark
            V (np array): covariance of noise for pixel coordinates, 4x4 matrix

        Returns:
            np array: kalman gain, 3M x 4N_t
        """
        assert V.shape == (4,4), "V must be 4x4"
        n_t = int(H.shape[0] / 4) # number of landmark seen in a frame
        IV = np.kron(np.eye(n_t), V) # 4N_t x 4N_t, noise
        temp = H @ cov @ H.T + IV
        return cov @ H.T @ np.linalg.inv(temp)

    @staticmethod
    def soft_max(x):
        temp = np.exp(x)
        return temp / temp.sum()
