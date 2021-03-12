import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

def load_data(file_name, load_features = False):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
        load_features: a boolean variable to indicate whether to load features 
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features. 4 = [leftx, lefty, rightx, righty]
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t. 3 = [x,y,z]
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t. 3 = [row, pitch, yaw]
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:

        t = data["time_stamps"] # time_stamps
        features = None 

        # only load features for 03.npz
        # 10.npz already contains feature tracks 
        if load_features:
            features = data["features"] # 4 x num_features : pixel coordinates of features
        
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame 
    
    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam


def visualize_trajectory_2d(pose,landmarks=None,path_name="trajectory",show_ori=False):
    '''
    function to visualize the trajectory in 2D. plot xy trajectory and orientation
    
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of pose, and each
                4*4 matrix is in SE(3)
        show_ori: show orientation of the robot, yaw angle
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name) # pose[0,3,:] -> x , pose[1,3,:] -> y
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    
    # plot landmarks
    if landmarks is not None:
        ax.scatter(landmarks[0], landmarks[1], 1, label="landmarks")

    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax
    
def calcualte_projection_derivative(q):
    assert len(q)==4, "q must be in R^4"
    derivative = np.zeros([4,4])
    q1,q2,q3,q4 = q[0],q[1],q[2],q[3]
    derivative[[0,1,3],[0,1,3]] = 1
    derivative[0,2] = -q1/q3
    derivative[1,2] = -q2/q3
    derivative[3,2] = -q4/q3
    derivative /= q3
    return derivative

def get_patch_idx(landmark_idxs):
    """Index corresponds to landmark seen in a frame.
    Used to retrieve or update landmarks shaped in 3M vector.

    Args:
        landmark_idxs (array): index of landmarks seen in a given frame

    Returns:
        array: indexes corresponds to landmark seen
    """
    idx_paches = [i for n in landmark_idxs for i in (3*n, 3*n+1, 3*n+2)]
    return np.array(idx_paches)

def calcualte_innovation(stero_cam, tf, world_T_imu, landmks_xyz, landmks_pixel_obs):
    landmk_pixel_pred = stero_cam.xyz_to_pixel(tf, world_T_imu, landmks_xyz)
    return landmks_pixel_obs - landmk_pixel_pred # 4 x N_t