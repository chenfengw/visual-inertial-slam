import numpy as np
import utils as ut


class KalmanFilter:
    def __init__(self):
        pass

    @staticmethod
    def _get_observation_jacobian_block(stero, tf, landmark_xyz, world_T_imu):
        optical_T_world = tf.optical_T_imu @ tf.inverse(world_T_imu)
        q = optical_T_world  @ tf.make_homogenous(landmark_xyz)
        proj_deriv = ut.calcualte_projection_derivative(q)

        return stero.M @ proj_deriv @ optical_T_world @ tf.P.T

    @staticmethod
    def calculate_observation_jacobian(stero, 
                                       tf,
                                       world_T_imu, 
                                       n_features, 
                                       landmark_idxs, 
                                       landmark_map):
        assert len(landmark_idxs) == n_features
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
    def soft_max(x):
        temp = np.exp(x)
        return temp / temp.sum()
