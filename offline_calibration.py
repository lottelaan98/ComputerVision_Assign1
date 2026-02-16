import cv2
import numpy as np
import glob

CHECKERBOARD_SIZE = (9, 6)    # inner corners (cols, rows)
SQUARE_SIZE = 0.018           

clicked_points = []


def create_object_points():
    """
    Creates 3D object points for the checkerboard in world coordinates.
    """
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0],
                           0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback to collect user clicks.
    """
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))


def get_manual_corners(image):
    """
    Allows the user to manually click the four outer checkerboard corners.
    """
    global clicked_points
    clicked_points = []

    clone = image.copy()

    cv2.namedWindow("Manual Corner Selection", cv2.WINDOW_NORMAL)

    cv2.setMouseCallback("Manual Corner Selection", mouse_callback)

    while True:
        display = clone.copy()
        for p in clicked_points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)

        cv2.imshow("Manual Corner Selection", display)

        if len(clicked_points) == 4:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    return np.array(clicked_points, dtype=np.float32)



def interpolate_corners(outer_corners):
    """
    Linearly interpolates all inner checkerboard corners from four outer corners.
    """
    tl, tr, br, bl = outer_corners
    cols, rows = CHECKERBOARD_SIZE
    corners = []

    for r in range(rows):
        alpha = r / (rows - 1)
        left = (1 - alpha) * tl + alpha * bl
        right = (1 - alpha) * tr + alpha * br

        for c in range(cols):
            beta = c / (cols - 1)
            point = (1 - beta) * left + beta * right
            corners.append(point)

    return np.array(corners, dtype=np.float32)


def get_image_corners(image):
    """
    Detects checkerboard corners automatically or manually if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)

    if ret:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        return corners.reshape(-1, 2), True

    outer = get_manual_corners(image)
    return interpolate_corners(outer), False


def calibrate_camera(objpoints, imgpoints, image_shape):
    """
    Performs geometric camera calibration using OpenCV.
    """
    flags = cv2.CALIB_ZERO_TANGENT_DIST
    return cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None, flags=flags
    )


def run_offline_calibration():
    """
    Runs all three required calibration experiments and saves the results.
    """
    images = glob.glob('*.jpeg')
    objp = create_object_points()

    objpoints, imgpoints, auto = [], [], []

    for fname in images:
        print(f"Processing {fname}...")
        img = cv2.imread(fname)
        corners, is_auto = get_image_corners(img)
        objpoints.append(objp)
        imgpoints.append(corners)
        auto.append(is_auto)

    shape = img.shape[1::-1]

    # ---------------- Run 1: all images ----------------
    ret1, K1, d1, rvecs1, tvecs1 = calibrate_camera(
        objpoints, imgpoints, shape
    )

    # Run 2: 5 auto + 5 manual
    auto_idx = [i for i, a in enumerate(auto) if a][:5]
    manual_idx = [i for i, a in enumerate(auto) if not a][:5]
    idx2 = auto_idx + manual_idx

    ret2, K2, d2, rvecs2, tvecs2 = calibrate_camera(
        [objpoints[i] for i in idx2],
        [imgpoints[i] for i in idx2],
        shape
    )

    # Run 3: only 5 automatic
    ret3, K3, d3, rvecs3, tvecs3 = calibrate_camera(
        [objpoints[i] for i in auto_idx],
        [imgpoints[i] for i in auto_idx],
        shape
    )

    np.savez("calibration_results.npz",
        # Run 1
        cameraMatrix_run1=K1,
        distCoeffs_run1=d1,
        rvecs_run1=rvecs1,
        tvecs_run1=tvecs1,

        # Run 2
        cameraMatrix_run2=K2,
        distCoeffs_run2=d2,
        rvecs_run2=rvecs2,
        tvecs_run2=tvecs2,

        # Run 3
        cameraMatrix_run3=K3,
        distCoeffs_run3=d3,
        rvecs_run3=rvecs3,
        tvecs_run3=tvecs3,
    )