import cv2
import numpy as np
import glob

IMAGE_PATH = "*.jpeg"
SHOW_IMAGES = True


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
    h = int(np.clip(179 * (1 - angle / 45.0), 0, 255))
    hsv = np.uint8([[[h, 255, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))  # convert to plain Python ints
    return color

def order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_text_onto_quad(image, quad_pts, text):
    quad = order_quad(quad_pts)

    W, H = 420, 160

    # 1Create white canvas
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    # Draw BLACK text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 7

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (W - tw) // 2
    y = (H + th) // 2

    cv2.putText(canvas, text, (x, y),
                font, font_scale, (0, 0, 0),
                thickness, cv2.LINE_AA)

    # Create mask where text exists (black pixels)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    text_mask = gray < 245   # detect dark pixels (the black text)

    # 4) Warp canvas and mask
    src = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    Hmat = cv2.getPerspectiveTransform(src, quad)

    warped_canvas = cv2.warpPerspective(canvas, Hmat,
                                        (image.shape[1], image.shape[0]))
    warped_mask = cv2.warpPerspective(text_mask.astype(np.uint8) * 255,
                                      Hmat,
                                      (image.shape[1], image.shape[0])) > 0

    # Overlay only the TEXT pixels (black text)
    image[warped_mask] = warped_canvas[warped_mask]



def quad_area(quad_pts):
    q = order_quad(quad_pts).astype(np.float32)
    return abs(cv2.contourArea(q))

def draw_cube_and_axes(image, rvec, tvec, K, d):
    """
    Draws 3D axes, cube, colored top plane, and distance annotation.
    """
    s = SQUARE_SIZE
    axes = np.float32([[0,0,0],[3*s,0,0],[0,3*s,0],[0,0,-3*s]])

    cube = np.float32([
        [0,0,0], [2*s,0,0], [2*s,2*s,0], [0,2*s,0],          # face A (z = 0)
        [0,0,-2*s], [2*s,0,-2*s], [2*s,2*s,-2*s], [0,2*s,-2*s]  # face B (z = -2s)
    ])

    img_axes = project(axes, rvec, tvec, K, d)
    img_cube = project(cube, rvec, tvec, K, d)

    pts = img_cube.astype(int)

    # Draw face A edges (indices 0..3)
    for i in range(4):
        cv2.line(image, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 0, 0), 2)

    # Draw face B edges (indices 4..7)
    for i in range(4, 8):
        cv2.line(image, tuple(pts[i]), tuple(pts[4 + (i + 1 - 4) % 4]), (0, 0, 0), 2)

    # Draw vertical edges
    for i in range(4):
        cv2.line(image, tuple(pts[i]), tuple(pts[i + 4]), (0, 0, 0), 2)

    # Draw axes
    o = tuple(img_axes[0].astype(int))
    cv2.line(image, o, tuple(img_axes[1].astype(int)), (0, 0, 255), 3)
    cv2.line(image, o, tuple(img_axes[2].astype(int)), (0, 255, 0), 3)
    cv2.line(image, o, tuple(img_axes[3].astype(int)), (255, 0, 0), 3)

    top = pts[4:8]  
    center_3d = np.mean(cube[4:8], axis=0)
    center_2d = np.mean(top, axis=0).astype(int)

    dist = compute_distance(center_3d, rvec, tvec)
    ang = compute_orientation(rvec)
    color = hsv_color(dist, ang)

    # Fill the chosen face
    cv2.fillConvexPoly(image, top, color)

    # Re-draw edges 
    for i in range(4, 8):
        cv2.line(image, tuple(pts[i]), tuple(pts[4 + (i + 1 - 4) % 4]), (0, 0, 0), 2)

    # Dot at center
    cv2.circle(image, tuple(center_2d), 6, (0, 0, 0), -1)


    text = f"{dist:.2f}m"
    if quad_area(top) > 1500:   
        warp_text_onto_quad(image, top, text)
    else:
        # fallback: always visible text
        cv2.putText(image, text, (center_2d[0] -70, center_2d[1] -20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3, cv2.LINE_AA)

    return image



if __name__ == "__main__":

    import os
    output_folder = "online_images_output"
    os.makedirs(output_folder, exist_ok=True)


    data = np.load("calibration_results.npz", allow_pickle=True)
    K = data["cameraMatrix_run1"]
    d = data["distCoeffs_run1"]

    CHECKERBOARD_SIZE = (9, 6)

    images = sorted(glob.glob(IMAGE_PATH))

    for fname in images:
        img = cv2.imread(fname)   
        if img is None:
            print("Could not read:", fname)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)
        if not found:
            print(fname, ": corners not found (skipping)") 
            continue

        corners = corners.reshape(-1, 2)

        objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE

        rvec, tvec = estimate_pose(objp, corners, K, d)

        out = draw_cube_and_axes(img.copy(), rvec, tvec, K, d)

        output_path = os.path.join(output_folder, fname)
        cv2.imwrite(output_path, out)
        print("Saved:", output_path)

        window_title = f"AR Test - {fname}"

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 800, 600)

        cv2.imshow(window_title, out)
        # press ESC to stop early
        if cv2.waitKey(0) & 0xFF == 27:  
            break

    cv2.destroyAllWindows()  

# if __name__ == "__main__":

#     data = np.load("calibration_results.npz", allow_pickle=True)
#     K = data["cameraMatrix_run1"]
#     d = data["distCoeffs_run1"]

#     CHECKERBOARD_SIZE = (9, 6)

#     fname = "image22.jpeg"
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)
#     if not found:
#         print("corners not found")
#         exit(0)

#     corners = corners.reshape(-1, 2)

#     objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
#     objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
#     objp *= SQUARE_SIZE

#     rvec, tvec = estimate_pose(objp, corners, K, d)
#     out = draw_cube_and_axes(img.copy(), rvec, tvec, K, d)

#     window_title = f"AR Test - {fname}"
#     cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(window_title, 800, 600)

#     cv2.imshow(window_title, out)
#     cv2.waitKey(0)
#     cv2.destroyWindow(window_title)

