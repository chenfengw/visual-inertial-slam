import numpy as np

class Transform():
    def __init__(self):
        self.imu_T_cam = np.array([[ 0.03717833, -0.09861822,  0.9944306 ,  1.5752681 ],
                                   [ 0.99926755, -0.00535534, -0.03789026,  0.00439141],
                                   [ 0.00906218,  0.99511094,  0.09834688, -0.65      ],
                                   [ 0.        ,  0.        ,  0.        ,  1.        ]])
        o_R_r = np.zeros([3,3])
        o_R_r[-1,0] = 1
        o_R_r[0,1] = -1
        o_R_r[1,-1] = -1
        self.optical_T_imu = np.linalg.inv(self.imu_T_cam)
        self.P = np.eye(3,4) # projection matrix, turn homogenous to normal 
        
    def optical_to_world(self,world_T_imu, optical):
        if Transform.is_3d(optical):
            optical = Transform.make_homogenous(optical)
        return world_T_imu @ self.imu_T_cam @ optical

    def world_to_optical(self,world_T_imu,world_xyz):
        if Transform.is_3d(world_xyz):
            world_xyz = Transform.make_homogenous(world_xyz)
        return Transform.inverse(self.imu_T_cam) @ Transform.inverse(world_T_imu) @ world_xyz

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
        assert R.shape == (3,3), "R must be 3x3"
        assert len(p) == 3, "p must be row vector of len 3"
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

    @staticmethod
    def inverse(T):
        """inverse pose matrix"""
        assert T.shape == (4,4), "T must be 4x4"
        return np.linalg.inv(T)

    @staticmethod
    def is_3d(x):
        if x.shape[0] == 3:
            return True
        else:
            return False

    @staticmethod
    def make_homogenous(x):
        """make xyz to be in homogenous coordinates, x: 3xn"""
        assert x.shape[0] == 3
        x = x.reshape(3,-1)
        n_col = x.shape[-1]
        return np.vstack((x,np.ones(n_col)))
    
    @staticmethod
    def adjoint_6d(u):
        assert len(u) == 6
        v = u[:3]
        w = u[3:]

        adjoint = np.zeros((6,6))
        w_hat = Transform.skew_3d(w)
        v_hat = Transform.skew_3d(v)
        adjoint[:3,:3] = w_hat
        adjoint[:3,3:] = v_hat
        adjoint[3:,3:] = w_hat
        return adjoint
    
    @staticmethod
    def circle_dot(s_h):
        assert len(s_h) == 4, "needs to be in homogenous"
        s = s_h[:3]
        result = np.zeros([4,6])
        result[:3,:3] = np.eye(3)
        result[:3,3:] = -Transform.skew_3d(s)
        return result