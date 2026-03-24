# SLAM launching after building container

## Main files

Container

- `/home/user/bridge/receiver_bridge_optim.py`

Server

- `SLAM/visual_sgraphs/launch/rgbd.launch.py` to launch SLAM

Orange Pi:

- `~/bridge/sender_direct.py`

## Launch order

### 1. Enter the container

```bash
docker start -ai <container_name>
```

### 2. Start receiver in the container

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/install/setup.bash

python3 /home/user/bridge/receiver_bridge_optim.py   --bind-address 0.0.0.0   --rgb-topic /camera/camera/color/image_raw   --depth-topic /camera/camera/aligned_depth_to_color/image_raw   --camera-info-topic /camera/camera/color/camera_info
```

### 3. Start sender on Orange Pi

```bash
python3 ~/bridge/sender_direct.py   --server-ip 192.168.50.185   --width 640   --height 480   --fps 15   --jpeg-quality 70
```

### 4. Check topics in the container

You may need to open another terminal and use

```bash
docker exec -it <container_name> bash
```

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
```

### 5. Start `vs_graphs` in the container

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

