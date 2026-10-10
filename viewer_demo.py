"""Explicitly simulated LiDAR fixtures for the local robot viewer, never live data."""
import math

import cv2
import numpy as np


def cup_points():
    points = []
    for z in np.linspace(-0.055, 0.055, 18):
        radius = 0.052 - (z + 0.055) / 0.11 * 0.017
        for angle in np.linspace(0, 2 * math.pi, 38, endpoint=False):
            points.append([round(radius * math.cos(angle), 4),
                           round(radius * math.sin(angle), 4), round(float(z), 4)])
    return points


def ball_points():
    return [[round(0.032 * math.sin(phi) * math.cos(theta), 4),
             round(0.032 * math.sin(phi) * math.sin(theta), 4),
             round(0.032 * math.cos(phi), 4)]
            for phi in np.linspace(0.1, math.pi - 0.1, 12)
            for theta in np.linspace(0, 2 * math.pi, 24, endpoint=False)]


CUP = cup_points()
BALL = ball_points()
TABLE = [[round(float(x), 4), round(float(y), 4), 0.699]
         for x in np.linspace(-0.2, 1.2, 36) for y in np.linspace(0.15, 1.4, 32)]


def synthetic(i, hz=5.0):
    t = i / hz
    phase = t % 30
    camera_position = np.array([0.47 + 0.025 * math.sin(t / 9), -0.21, 1.12])
    forward = np.array([0.045 * math.sin(t / 7), 0.87, -0.49])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.column_stack((right, down, forward, camera_position))
    objects, balls = [], []
    for k, (x, y, color) in enumerate([(0.18, 0.61, "green"), (0.47, 0.77, "yellow"), (0.77, 0.61, "purple")]):
        x += 0.018 * math.sin(t / 3 + k)
        occluded = k == 2 and 9 < phase < 23
        state = "OCCLUDED" if occluded else "VISIBLE"
        objects.append({
            "id": k, "label": "solo cup", "target": k == 1, "state": state,
            "position": [x, y, 0.755], "velocity": [0.006 * math.cos(t / 3 + k), 0, 0],
            "size": [0.104, 0.104, 0.11], "last_seen_t": t - (phase - 9) if occluded else t,
            "since_seen_s": phase - 9 if occluded else 0, "confidence": 1.0,
            "detection_score": None if occluded else 0.91, "reacquired": False,
            "has_ball": phase > 5, "balls_inside": [f"{color} ball"] if phase > 5 else [],
            "point_cloud": {"frame": "object", "points": CUP},
        })
        balls.append({
            "id": f"{color} ball", "label": "ball", "color": color,
            "state": "INSIDE" if phase > 5 else "VISIBLE", "inside": k if phase > 5 else None,
            "position": [x, y if phase > 5 else y - 0.18, 0.732], "size": [0.064] * 3,
            "since_seen_s": max(0, phase - 5), "detection_score": None if phase > 5 else 0.8,
            "verdict": None, "point_cloud": {"frame": "object", "points": BALL},
        })
    occluders = [{"id": "back-wall", "label": "wall", "position": [0.5, 1.42, 0.98], "size": [1.4, 0.025, 0.56]}]
    if 9 < phase < 23:
        occluders.append({"id": "cover", "label": "box", "position": [objects[2]["position"][0], 0.60, 0.79], "size": [0.19, 0.20, 0.18]})
    return {"source": "synthetic", "t": round(t, 3), "frame": i, "target_id": 1,
            "phase": "track", "fps": hz,
            "camera": {"position": camera_position.tolist(), "forward": forward.tolist(),
                       "T_world_cam": transform.reshape(-1).tolist()},
            "objects": objects, "ball": {**balls[1], "balls": balls},
            "scene": {"floor_z": 0.699, "point_cloud": {"frame": "world", "points": TABLE}},
            "occluders": occluders}


def render_frame(state):
    """Render the same simulated world through the reported OpenCV camera pose."""
    image = np.full((720, 960, 3), (22, 25, 24), dtype=np.uint8)
    matrix = np.asarray(state["camera"]["T_world_cam"]).reshape(3, 4)
    rotation, origin = matrix[:, :3], matrix[:, 3]

    def project(points):
        camera = (np.asarray(points) - origin) @ rotation
        depth = camera[:, 2]
        pixel = camera[:, :2] / np.maximum(depth[:, None], 0.01) * 554 + [480, 360]
        return pixel.astype(int), depth

    def cloud(points, color, radius=1):
        pixel, depth = project(points)
        for idx in np.argsort(-depth):
            x, y = pixel[idx]
            if depth[idx] > 0.01 and 0 <= x < 960 and 0 <= y < 720:
                cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    cloud(TABLE, (78, 85, 80))
    for obj in state["objects"]:
        cloud(np.asarray(CUP) + obj["position"], (92, 105, 210), 2)
    for obj in state["occluders"]:
        center, half = np.asarray(obj["position"]), np.asarray(obj["size"]) / 2
        corners = [center + half * [x, y, z] for z in [-1, 1] for y in [-1, 1] for x in [-1, 1]]
        pixel, depth = project(corners)
        if np.all(depth > 0):
            hull = cv2.convexHull(pixel.astype(np.int32))
            cv2.fillConvexPoly(image, hull, (51, 58, 57), cv2.LINE_AA)
            cv2.polylines(image, [hull], True, (105, 121, 113), 1, cv2.LINE_AA)
    for obj in state["objects"]:
        # X-ray points remain visible through the simulated box.
        cloud(np.asarray(CUP)[::3] + obj["position"], (194, 207, 93), 1)
    bgr = {"yellow": (108, 208, 244), "green": (129, 218, 147), "purple": (232, 152, 206)}
    for ball in state["ball"]["balls"]:
        cloud(np.asarray(BALL) + ball["position"], bgr[ball["color"]], 2)
    cv2.putText(image, "SIMULATED RGB + LIDAR", (26, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (169, 186, 180), 1, cv2.LINE_AA)
    cv2.putText(image, f'FRAME {state["frame"]:05d}', (26, 688), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 165, 159), 1, cv2.LINE_AA)
    return image
