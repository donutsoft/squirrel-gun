from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics.
    fx, fy: focal lengths in pixels
    cx, cy: principal point in pixels
    """
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_fov(image_width: int, image_height: int, hfov_deg: float, vfov_deg: Optional[float] = None) -> "CameraIntrinsics":
        """Build intrinsics from field-of-view and image size.
        If vfov_deg is not provided, it is derived from aspect ratio to keep square pixels.
        """
        hfov = np.radians(hfov_deg)
        fx = (image_width / 2.0) / np.tan(hfov / 2.0)
        if vfov_deg is None:
            # Assume square pixels: fy/height == fx/width
            fy = fx * (image_height / image_width)
        else:
            vfov = np.radians(vfov_deg)
            fy = (image_height / 2.0) / np.tan(vfov / 2.0)
        cx = image_width / 2.0
        cy = image_height / 2.0
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

def pixel_to_angles(u: float, v: float, K: CameraIntrinsics) -> Tuple[float, float]:
    """Convert a pixel (u, v) to yaw/pitch angles (radians) relative to the optical axis.

    - u, v: pixel coordinates (origin at top-left, v increases downward)
    - K: camera intrinsics

    Returns (yaw, pitch):
      yaw > 0 means turn right (positive X), pan 
      pitch > 0 means turn up (negative image Y). tilt
    """
    x = (u - K.cx) / K.fx  # normalized image plane x (at z=1)
    y = (v - K.cy) / K.fy  # normalized image plane y (at z=1), positive down

    # Direction vector d = (x, y, 1). Yaw about vertical axis from +Z toward +X.
    yaw = float(np.arctan2(x, 1.0))
    # Pitch about horizontal axis from +Z toward -Y (up). Negative image y is up.
    pitch = float(np.arctan2(-y, 1.0))
    return yaw, pitch

def pixel_to_plane_point(u: float, v: float, Z: float, K: CameraIntrinsics) -> Tuple[float, float]:
    """Project pixel (u, v) to a point (X, Y) on a plane at depth Z along the camera Z axis.
    Assumes standard pinhole model with camera at origin, looking along +Z, with pixels
    mapped to normalized coordinates by (u - cx)/fx and (v - cy)/fy.
    """
    x = (u - K.cx) / K.fx
    y = (v - K.cy) / K.fy
    X = x * Z
    Y = y * Z
    return X, Y

def aim_angles_from_pixel(
    u: float,
    v: float,
    Z: float,
    K: CameraIntrinsics,
    pivot_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float]:
    """Compute pan/tilt angles (yaw, pitch) to aim a hose at the 3D point where the camera ray through (u, v)
    intersects the plane z=Z (camera coordinates). Optionally account for a pivot offset.

    - u, v: pixel coordinates (top-left origin)
    - Z: distance to the target plane along camera Z axis (same units as your rig)
    - K: camera intrinsics
    - pivot_offset: (dx, dy, dz) offset from camera center to the hose pivot, in camera coords.
      For example, dx > 0 means the pivot is to the right of the camera by dx units.

    Returns (yaw, pitch) in radians.

    Notes:
    - If the hose pivots at the camera center, pivot_offset = (0, 0, 0) and the angles are simply the
      direction angles to the intersection point. When pivot is offset, we aim from the pivot to the point.
    """
    X, Y = pixel_to_plane_point(u, v, Z, K)
    # Vector from pivot to target point on plane
    dx, dy, dz = pivot_offset
    vx = X - dx
    vy = Y - dy
    vz = Z - dz
    # Yaw: rotation around +Y (up), from +Z toward +X
    yaw = float(np.arctan2(vx, vz))
    # Pitch: rotation around +X (right), from +Z toward -Y (up)
    # Positive pitch aims upward; image y down is negative pitch.
    hyp = float(np.hypot(vx, vz))
    pitch = float(np.arctan2(-vy, hyp))
    return yaw, pitch

def degrees(rad: float) -> float:
    return float(np.degrees(rad))

def radians(deg: float) -> float:
    return float(np.radians(deg))

def aim_angles_left_offset(
    u: float,
    v: float,
    Z: float,
    K: CameraIntrinsics,
    left_offset: float,
    up_offset: float = 0.0,
    forward_offset: float = 0.0,
) -> Tuple[float, float]:
    """Compute aim angles when the hose pivot is a fixed distance to the LEFT of the camera.
    Coordinate convention (camera frame):
    - +X to the RIGHT; LEFT is -X
    - +Y UP; -Y down
    - +Z forward (optical axis)

    left_offset is a positive distance to the camera's left side, so the pivot's X offset is -left_offset.
    """
    dx = -abs(left_offset)  # left of the camera is negative X
    dy = up_offset
    dz = forward_offset
    return aim_angles_from_pixel(u, v, Z, K, pivot_offset=(dx, dy, dz))

def calculate_iteratively_pitch_yaw() -> List[Tuple[int, int, float, float]]:
    image_width = 1280
    image_height = 720
    K = CameraIntrinsics.from_fov(image_width=image_width, image_height=image_height, hfov_deg=98.0)
    Z = 1.277  # meters to the far plane
    results = []
    for y in range(image_width):
        for x in range(image_height):

            yaw, pitch = aim_angles_from_pixel(y, x, Z, K)
            point = (y,x, yaw, pitch)
            results.append(point)
    
    return results

def find_tuple_by_yx(y: int, x: int, items: List[Tuple[int, int, float, float]]) -> Optional[Tuple[int, int, float, float]]:
    for t in items:
        if t[0] == y and t[1] == x:
            return t
    return None

def get_pitch_yaw_from_tuple(self, x, y):
    results = []
    results = calculate_iteratively_pitch_yaw()
    #print(results)
    t = find_tuple_by_yx(y, x, results)
    print(t)
    print(t[2], t[3])
    
    return (t[2], t[3])
    
#get_pitch_yaw_from_tuple()