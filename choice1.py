import cv2
import numpy as np

from offline_calibration import create_object_points
from online_ar import estimate_pose, draw_cube_and_axes

CHECKERBOARD_SIZE = (9, 6)

def main():
    # 1) Load camera calibration (intrinsics K and distortion d)
    data = np.load("calibration_results.npz")
    K = data["cameraMatrix_run1"]
    d = data["distCoeffs_run1"]

    # 2) Prepare the 3D checkerboard points (same for every frame)
    objp = create_object_points()

    # 3) Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    # 4) Read one frame to get width/height for the video writer
    ret, frame = cap.read()

    h, w = frame.shape[:2]

    # 5) Create a video writer that saves a .mov file
    #    'mp4v' often works with .mov container on many systems
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("online_phase_output.mov", fourcc, 30.0, (w, h))

    while True:
        # 6) Grab a frame
        ret, frame = cap.read()
        if not ret:
            break

        # 7) Convert to gray for corner detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 8) Detect checkerboard corners
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)

        # 9) If found: refine corners, estimate pose, draw AR overlay
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            corners = corners.reshape(-1, 2)

            rvec, tvec = estimate_pose(objp, corners, K, d)
            frame = draw_cube_and_axes(frame, rvec, tvec, K, d)

        # 10) Show the result
        cv2.imshow("Webcam AR", frame)

        # 11) Save the frame into the .mov video
        out.write(frame)

        # 12) Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 13) Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
