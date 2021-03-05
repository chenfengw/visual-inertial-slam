import numpy as np

class Transform():
    def __init__(self,imu_T_cam):
        self.imu_T_cam = imu_T_cam
        o_R_r = np.zeros([3,3])
        o_R_r[-1,0] = 1
        o_R_r[0,1] = -1
        o_R_r[1,-1] = -1
        self.optical_T_cam = Transform.calcualte_pose(o_R_r, np.zeros(3))
        self.cam_T_optical = Transform.calcualte_pose(o_R_r.T, np.zeros(3))
    
    def optical_to_world(self,world_T_imu, optical):
        return world_T_imu @ self.imu_T_cam @ self.cam_T_optical @ optical

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