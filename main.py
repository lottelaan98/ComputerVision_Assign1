import cv2
import numpy as np
from offline_calibration import run_offline_calibration, create_object_points
from online_ar import estimate_pose, draw_cube_and_axes

CHECKERBOARD_SIZE = (9, 6)

def main():
    """
    Main entry point for offline calibration and online AR visualization.
    """
    # run_offline_calibration()

    data = np.load("calibration_results.npz")

    K = data["cameraMatrix_run1"]
    d = data["distCoeffs_run1"]
    rvecs = data["rvecs_run1"]
    tvecs = data["tvecs_run1"]


    print(K)
    print(d)
    print(rvecs)
    print(tvecs)

    # moet nieuwe testimage zien? 
    test_img = cv2.imread("image27.jpeg")

    objp = create_object_points()

    # corners_test must be detected automatically
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    _, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)
    corners = corners.reshape(-1, 2)


    rvec, tvec = estimate_pose(objp, corners, K, d)
    result_img = draw_cube_and_axes(test_img.copy(), rvec, tvec, K, d)

    # ---------------- Show result ----------------
    cv2.namedWindow("AR Result", cv2.WINDOW_NORMAL)
    cv2.imshow("AR Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    data = np.load("calibration_results.npz", allow_pickle=True)

    for i in [1, 2, 3]:
        K = data[f"cameraMatrix_run{i}"]
        d = data[f"distCoeffs_run{i}"]

        rvec, tvec = estimate_pose(objp, corners, K, d)
        out = draw_cube_and_axes(test_img.copy(), rvec, tvec, K, d)


        window = cv2.imshow(f"Run {i}", out)

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 800, 600)   # <-- change size here
        cv2.imshow(window, out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
