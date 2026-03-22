import argparse
import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
P0_MOVING_AVG_WINDOW = 10
MOVE_XY_THRESHOLD = 0.03
MOVE_Z_THRESHOLD = 0.02
TURN_DEAD_ZONE_DEG = 10


def get_depth_at_pixel(depth_image, depth_scale, px, py):
    """Return distance in meters for a pixel, or None when depth is invalid."""
    height, width = depth_image.shape
    px = min(max(px, 0), width - 1)
    py = min(max(py, 0), height - 1)
    depth_raw = int(depth_image[py, px])
    if depth_raw <= 0:
        return None
    return depth_raw * depth_scale


def get_movement_directions(current_xyz, origin_xyz):
    """Return movement directions relative to the origin point."""
    cur_x, cur_y, cur_z = current_xyz
    org_x, org_y, org_z = origin_xyz
    directions = []

    if cur_y < org_y - MOVE_XY_THRESHOLD:
        directions.append("up")
    elif cur_y > org_y + MOVE_XY_THRESHOLD:
        directions.append("down")

    if cur_x < org_x - MOVE_XY_THRESHOLD:
        directions.append("left")
    elif cur_x > org_x + MOVE_XY_THRESHOLD:
        directions.append("right")

    if cur_z < org_z - MOVE_Z_THRESHOLD:
        directions.append("forward")
    elif cur_z > org_z + MOVE_Z_THRESHOLD:
        directions.append("backward")

    return directions


def draw_hand_skeleton(image_bgr, hand_landmarks_list):
    """Draw hand skeleton for all detected hands."""
    h, w, _ = image_bgr.shape
    for hand_landmarks in hand_landmarks_list:
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(image_bgr, (x, y), 3, (0, 255, 255), -1)

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(
                image_bgr,
                points[start_idx],
                points[end_idx],
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )


def draw_gesture_labels(
    image_bgr,
    recognition_result,
    depth_image,
    depth_scale,
    p0_histories,
    control_start_points,
):
    """Draw gesture labels near each detected hand."""
    h, w, _ = image_bgr.shape

    for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
        if i >= len(p0_histories):
            p0_histories.append(deque(maxlen=P0_MOVING_AVG_WINDOW))
        if i >= len(control_start_points):
            control_start_points.append(None)

        x0 = hand_landmarks[0].x
        y0 = hand_landmarks[0].y
        px0 = int(x0 * w)
        py0 = int(y0 * h)
        z0_m = get_depth_at_pixel(depth_image, depth_scale, px0, py0)
        if z0_m is None:
            continue

        p0_histories[i].append((x0, y0, z0_m))
        avg_x0 = sum(v[0] for v in p0_histories[i]) / len(p0_histories[i])
        avg_y0 = sum(v[1] for v in p0_histories[i]) / len(p0_histories[i])
        avg_z0 = sum(v[2] for v in p0_histories[i]) / len(p0_histories[i])

        gesture_name = "Unknown"
        gesture_score = 0.0
        gesture_lines = ["Unknown"]
        if i < len(recognition_result.gestures) and recognition_result.gestures[i]:
            top_gesture = recognition_result.gestures[i][0]
            gesture_lines = [
                f"{gesture.category_name}: {gesture.score:.2f}"
                for gesture in recognition_result.gestures[i]
            ]
            gesture_name = top_gesture.category_name
            gesture_score = top_gesture.score

        control_detected = gesture_name == "Closed_Fist"
        if control_detected:
            if control_start_points[i] is None:
                control_start_points[i] = (avg_x0, avg_y0, avg_z0)
        else:
            control_start_points[i] = None

        handedness = ""
        if i < len(recognition_result.handedness) and recognition_result.handedness[i]:
            handedness = recognition_result.handedness[i][0].category_name

        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]

        # print(f"Gesture: {gesture_name}, {hand_landmarks}")
        text_x = int(min(xs) * w)
        text_y = int(min(ys) * h) - 10
        text_y = max(text_y, 25)

        label = f"{gesture_name} ({gesture_score:.2f})"
        if handedness:
            label = f"{label} [{handedness}]"

        cv2.putText(
            image_bgr,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        for line_index, gesture_line in enumerate(gesture_lines, start=1):
            cv2.putText(
                image_bgr,
                gesture_line,
                (text_x, text_y + 26 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (120, 255, 120),
                2,
                cv2.LINE_AA,
            )
            print(f"  {gesture_line}")

        info_y = text_y + 26 * (len(gesture_lines) + 1)
        cv2.putText(
            image_bgr,
            f"p0(avg10) xyz: {avg_x0:.2f}, {avg_y0:.2f}, {avg_z0:.2f} m",
            (text_x, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if control_start_points[i] is not None:
            start_x, start_y, start_z = control_start_points[i]
            start_px = int(start_x * w)
            start_py = int(start_y * h)
            cv2.circle(image_bgr, (start_px, start_py), 6, (0, 0, 255), -1)

            directions = get_movement_directions(
                (avg_x0, avg_y0, avg_z0), (start_x, start_y, start_z)
            )
            if directions:
                cv2.putText(
                    image_bgr,
                    " / ".join(directions),
                    (text_x, info_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        if gesture_name == "Open_Palm" and len(hand_landmarks) > 12:
            x12 = hand_landmarks[12].x
            y12 = hand_landmarks[12].y

            p0 = (int(x0 * w), int(y0 * h))
            p0_top = (p0[0], 0)
            p12 = (int(x12 * w), int(y12 * h))
            up_len = max(p0[1], 1)
            zone_dx = int(up_len * math.tan(math.radians(TURN_DEAD_ZONE_DEG)))
            left_zone_top = (max(p0[0] - zone_dx, 0), 0)
            right_zone_top = (min(p0[0] + zone_dx, w - 1), 0)

            cv2.line(image_bgr, p0, p0_top, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.line(image_bgr, p0, left_zone_top, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.line(image_bgr, p0, right_zone_top, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.line(image_bgr, p0, p12, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(image_bgr, p12, 5, (255, 255, 0), -1)

            dx = x12 - x0
            dy = y12 - y0
            angle_from_center_deg = math.degrees(math.atan2(dx, -dy))
            if abs(angle_from_center_deg) <= TURN_DEAD_ZONE_DEG:
                turn_text = "No Turn"
            elif dx < 0:
                turn_text = "Turn Right"
            else:
                turn_text = "Turn Left"
            cv2.putText(
                image_bgr,
                turn_text,
                (text_x, info_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    for i in range(len(recognition_result.hand_landmarks), len(control_start_points)):
        control_start_points[i] = None


def run_realtime_gesture_detection(model_path, camera_id, show_depth):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)

    last_timestamp_ms = 0
    p0_histories = [deque(maxlen=P0_MOVING_AVG_WINDOW) for _ in range(2)]
    control_start_points = [None for _ in range(2)]
    try:
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale}")

        for _ in range(10):
            pipeline.wait_for_frames(5000)

        while True:
            frames = pipeline.wait_for_frames(5000)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            frame_bgr = cv2.flip(frame_bgr, 1)
            depth_image = cv2.flip(depth_image, 1)
            frame_depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                draw_hand_skeleton(frame_bgr, result.hand_landmarks)
                draw_gesture_labels(
                    frame_bgr,
                    result,
                    depth_image,
                    depth_scale,
                    p0_histories,
                    control_start_points,
                )

            cv2.imshow("Real-time Gesture Detection", frame_bgr)
            if show_depth:
                cv2.imshow("Depth", frame_depth_colormap)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        recognizer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time hand gesture detector")
    parser.add_argument(
        "--model",
        type=str,
        default="gesture_recognizer.task",
        help="Path to MediaPipe gesture recognizer task model",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device id (default: 0)",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show the depth colormap window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_realtime_gesture_detection(args.model, args.camera_id, args.show_depth)
