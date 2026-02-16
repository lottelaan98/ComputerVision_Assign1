import cv2
import numpy as np
from offline_calibration import run_offline_calibration, create_object_points
from online_ar import estimate_pose, draw_cube_and_axes

CHECKERBOARD_SIZE = (9, 6)

def main():
    """
    Main entry point for offline calibration and online AR visualization.
    """
    run_offline_calibration()

    data = np.load("calibration_results.npz", allow_pickle=True)

    # moet nog testimage maken
    test_img = cv2.imread("image1.jpg")

    objp = create_object_points()

    # # corners_test MUST be detected automatically
    # gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    # _, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)
    # corners = corners.reshape(-1, 2)

    # for i, run in enumerate(["run1", "run2", "run3"], 1):
    #     _, K, d, _, _ = data[run]
    #     rvec, tvec = estimate_pose(objp, corners, K, d)
    #     out = draw_cube_and_axes(test_img.copy(), rvec, tvec, K, d)

    #     cv2.imshow(f"Run {i}", out)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
