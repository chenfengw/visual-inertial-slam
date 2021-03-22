# Visual Inertial SLAM 
Simultaneous localization and mapping (SLAM) is the problem of localizing the robot and creating a map of the environment at the same time. This project contains code for visual inertial SLAM algorithm using Extended Kalman Filter.

## Demo
![SLAM](figs/slam.gif)

## Requirements
To install requirements:
```
pip install -r requirements.txt
```

## SLAM
To perform SLAM, run this script:
```
python slam_main.py
```
> recommend to run this script in Python interactive.

## Sensors
- Stereo camera
- IMU

## Files Directory
- [kalman_filter.py](kalman_filter.py)
  - calculating jacobain, kalman gain
- [mapping.py](mapping.py)
  - landmark and pose estimation
- [sensors.py](sensors.py)
  - IMU and stereo camera
- [transform.py](transform.py)
  - hat map, transformation between frames