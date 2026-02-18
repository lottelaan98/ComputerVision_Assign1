import cv2
import numpy as np
import glob
import re
import matplotlib.pyplot as plt

CHECKERBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.018  # meters

def create_object_points():
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp

def numeric_sort(files):
    return sorted(files, key=lambda x: int(re.search(r"\d+", x).group()))

def main():
    # Load calibration intrinsics and distortion
    data = np.load("calibration_results.npz", allow_pickle=True)
    K = data["cameraMatrix_run1"]
    d = data["distCoeffs_run1"]

    # Prepare the chessboard 3D points (world frame = checkerboard frame)
    objp = create_object_points()

    # Collect camera centers for each image
    cam_centers = []
    labels = []

    images = numeric_sort(glob.glob("*.jpeg"))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect corners
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)
        if not found:
            print(f"{fname}: corners not found (skipping)")
            continue

        # Refine corners for more stable pose
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # Estimate pose (rvec, tvec) that maps world points -> camera points
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, d)
        if not ok:
            print(f"{fname}: solvePnP failed (skipping)")
            continue

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Camera center in world coordinates:
        # X_cam = R * X_world + t  then  camera center C_world = -R^T * t
        C = -R.T @ tvec

        cam_centers.append(C.reshape(3))
        labels.append(fname)

    cam_centers = np.array(cam_centers)  # shape (N, 3)

    if len(cam_centers) == 0:
        print("No camera poses computed. Check that corners are detected in some images.")
        return

    # Print a quick reflection / interpretation
    dists = np.linalg.norm(cam_centers, axis=1)
    print("\n=== Pose summary ===")
    print("Num poses:", len(cam_centers))
    print("Distance to board (min/mean/max) meters:",
          f"{dists.min():.2f} / {dists.mean():.2f} / {dists.max():.2f}")

    # 3D plot of camera centers
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2])

    ax.set_title("Camera positions relative to chessboard (world frame)")
    ax.set_xlabel("X (m) along board")
    ax.set_ylabel("Y (m) along board")
    ax.set_zlabel("Z (m) board normal")

    # Save plot
    plt.tight_layout()
    plt.savefig("camera_positions_3d.png", dpi=200)
    plt.show()

    print("\nSaved plot to: camera_positions_3d.png")

if __name__ == "__main__":
    main()
