import cv2
import numpy as np

SQUARE_SIZE = 0.018

def estimate_pose(objpoints, imgpoints, K, d):
    """
    Estimates camera pose using solvePnP.
    """
    _, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, K, d)
    return rvec, tvec


def project(points, rvec, tvec, K, d):
    """
    Projects 3D points to image coordinates.
    """
    imgpts, _ = cv2.projectPoints(points, rvec, tvec, K, d)
    return imgpts.reshape(-1, 2)


def compute_distance(point_3d, rvec, tvec):
    """
    Computes distance from a 3D point to the camera.
    """
    R, _ = cv2.Rodrigues(rvec)
    cam_pt = R @ point_3d.reshape(3, 1) + tvec
    return np.linalg.norm(cam_pt)


def compute_orientation(rvec):
    """
    Computes orientation angle between top plane and camera view direction.
    """
    R, _ = cv2.Rodrigues(rvec)
    normal = R @ np.array([0, 0, 1])
    cos_angle = normal[2] / np.linalg.norm(normal)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def hsv_color(distance, angle):
    """
    Computes HSV color based on distance and orientation rules.
    """
    v = int(np.clip(255 * (1 - distance / 4.0), 0, 255))
    h = int(np.clip(255 * (1 - angle / 45.0), 0, 255))
    hsv = np.uint8([[[h, 255, v]]])
    return tuple(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])


def draw_cube_and_axes(image, rvec, tvec, K, d):
    """
    Draws 3D axes, cube, colored top plane, and distance annotation.
    """
    s = SQUARE_SIZE
    axes = np.float32([[0,0,0],[3*s,0,0],[0,3*s,0],[0,0,-3*s]])
    cube = np.float32([
        [-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0],
        [-s,-s,-s],[s,-s,-s],[s,s,-s],[-s,s,-s]
    ])

    img_axes = project(axes, rvec, tvec, K, d)
    img_cube = project(cube, rvec, tvec, K, d)

    o = tuple(img_axes[0].astype(int))
    cv2.line(image, o, tuple(img_axes[1].astype(int)), (0,0,255), 3)
    cv2.line(image, o, tuple(img_axes[2].astype(int)), (0,255,0), 3)
    cv2.line(image, o, tuple(img_axes[3].astype(int)), (255,0,0), 3)

    top = img_cube[4:].astype(int)
    center_3d = np.mean(cube[4:], axis=0)
    center_2d = np.mean(top, axis=0).astype(int)

    dist = compute_distance(center_3d, rvec, tvec)
    ang = compute_orientation(rvec)
    color = hsv_color(dist, ang)

    cv2.fillConvexPoly(image, top, color)
    cv2.circle(image, tuple(center_2d), 5, (0,0,0), -1)
    cv2.putText(image, f"{dist:.2f} m", tuple(center_2d+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    return image
