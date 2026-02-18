import cv2
import numpy as np
from offline_calibration import run_offline_calibration, create_object_points
from online_ar import estimate_pose, draw_cube_and_axes

CHECKERBOARD_SIZE = (9, 6)

def main():
    """
    Main entry point for offline calibration and online AR visualization.
    """

    # Load calibration results
    data = np.load("calibration_results.npz", allow_pickle=True)

    # Load test image (must be automatically detected)
    test_img = cv2.imread("image38.jpeg")
    if test_img is None:
        raise IOError("Could not load test image")

    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)

    if not found:
        raise RuntimeError("Chessboard corners NOT found in test image")

    corners = corners.reshape(-1, 2)
    objp = create_object_points()

    # Loop over the 3 runs
    for i in [1, 2, 3]:
        K = data[f"cameraMatrix_run{i}"]
        d = data[f"distCoeffs_run{i}"]

        rvec, tvec = estimate_pose(objp, corners, K, d)
        out = draw_cube_and_axes(test_img.copy(), rvec, tvec, K, d)

        filename = f"run{i}_test_cube.png"
        cv2.imwrite(filename, out)
        print(f"Saved {filename}")

        window_name = f"Run {i}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.imshow(window_name, out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

