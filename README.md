# SLAM launching after building container

<<<<<<< HEAD
conda env create -f environment.yml
=======
## Main files

Container

- `/home/user/bridge/receiver_bridge.py` 
- `/home/user/bridge/run_receiver.sh` to launch `receiver_bridge.py`

Server

- `SLAM/visual_sgraphs/launch/rgbd.launch.py` to launch SLAM

Orange Pi:

- `~/bridge/sender_bridge.py`
- `~/scripts/run_sender.py` to launch `sender_bridge.py`
- `~/scripts/run_realsense.sh` to run camera
- `~/scripts/run_realsense_imu.sh` to rum camera with IMU

## Launch order

### 1. Enter the container

```bash
docker start -ai <container_name>
```

### 2. Start receiver in the container

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/install/setup.bash
```
`~/bridge/run_receiver.sh`

### 3. Start RealSense on Orange Pi

```bash
source /opt/ros/humble/setup.bash
```
`~/scripts/run_realsense.sh` 

### 4. Start sender on Orange Pi

`~/scripts/run_sender.py`

### 5. Check topics in the container

You may need to open another terminal and use

```bash
docker exec -it <container_name> bash
```

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
```

### 6. Start `vs_graphs` in the container

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/install/setup.bash
export DISPLAY=:1

ros2 launch vs_graphs rgbd.launch.py \
  offline:=false \
  launch_rviz:=true \
  colored_pointcloud:=true \
  visualize_segmented_scene:=true \
  sensor_config:=RealSense_D435i \
  rgb_image_topic:=/camera/camera/color/image_raw \
  depth_image_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  rgb_camera_info_topic:=/camera/camera/color/camera_info
```

## Quick checks

Tracking:

```bash
ros2 topic hz /vs_graphs/tracking_image
ros2 topic info /vs_graphs/tracked_points
```

Semantics:

```bash
ros2 node info /segmenter_ros
ros2 topic hz /vs_graphs/keyframe_image
ros2 topic hz /camera/color/image_segment
```
>>>>>>> 253783a (SLAM)
