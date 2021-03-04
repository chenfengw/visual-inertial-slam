import numpy as np

class Transform():
    def __init__(self):
        R_lidar = np.array([[ 1.30201e-03,  7.96097e-01,  6.05167e-01],
                            [ 9.99999e-01, -4.19027e-04, -1.60026e-03],
                            [-1.02038e-03,  6.05169e-01, -7.96097e-01]])
        p_lidar = np.array([0.8349, -0.0126869, 1.76416])
        self.car_T_lidar = self.calcualte_pose(R_lidar,p_lidar)

    @staticmethod
    def skew_3d(vector):
        """
        this function returns a numpy array with the skew symmetric cross product matrix for vector.
        the skew symmetric cross product matrix is defined such that
        np.cross(a, b) = np.dot(skew(a), b)

        :param vector: An array like vector to create the skew symmetric cross product matrix for
        :return: A numpy array of the skew symmetric cross product vector
        """
        assert len(vector) == 3
        return np.array([[0, -vector.item(2), vector.item(1)],
                        [vector.item(2), 0, -vector.item(0)],
                        [-vector.item(1), vector.item(0), 0]])
    
    @staticmethod
    def calculate_twist(vector):
        """Calculating hat map for a 6d vector

        Args:
            vector (np array): in R^6, [linear_velocity (x,y,z), angular Velcro (row,pitch,yaw)]

        Returns:
            np array: twist matrix 4x4
        """
        assert len(vector) == 6
        p = vector[:3]
        theta = vector[-3:]
        theta_hat = Transform.skew_3d(theta)

        # create twist matrix
        twist = np.zeros([4,4])
        twist[:3,:3] = theta_hat
        twist[:3,-1] = p
        return twist
        
    @staticmethod
    def calcualte_pose(R,p):
        # assert R.shape == (3,3), "R must be 3x3"
        # assert len(p) == 3, "p must be row vector of len 3"
        T = np.zeros([4,4])
        T[:3,:3] = R
        T[:3,-1] = p
        T[-1,-1] = 1
        return T

    @staticmethod
    def get_rotation_matrix(theta):
        """Calculate rotation matrix around z axis rotated by angle theta

        Args:
            theta (float): rotation angle in radians

        Returns:
            np array: 3x3 rotation matrix
        """
        R = np.zeros([3,3])
        R_2d = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
        R[:2,:2] = R_2d
        R[-1,-1] = 1
        return R