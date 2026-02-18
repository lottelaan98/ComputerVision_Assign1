import cv2
import numpy as np
import glob
import re

CHECKERBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.018

def create_object_points():
    """
    Creates object points for the checkerboard pattern.
    """
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp

def detect_corners_all_images(image_glob="*.jpeg"):
    """
    Detects chessboard corners in all images matching the glob pattern.
    """
    images = sorted(glob.glob(image_glob), key=lambda x: int(re.search(r"\d+", x).group()))
    objp = create_object_points()

    objpoints, imgpoints, used_files = [], [], []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            print(f"{fname}: detection failed (skipping)")
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        objpoints.append(objp)
        imgpoints.append(corners)
        used_files.append(fname)

    image_shape = gray.shape[::-1]  # (w, h)
    return objpoints, imgpoints, used_files, image_shape

def per_image_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, d):
    """
    Computes mean reprojection error for each image.
    """
    errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, d)
        proj = proj.reshape(-1, 2)
        obs = imgpoints[i].reshape(-1, 2)

        # mean pixel error for this image
        err = np.mean(np.linalg.norm(proj - obs, axis=1))
        errors.append(err)
    return errors

def iterative_reject_calibration(objpoints, imgpoints, files, image_shape,
                                 max_mean_err=0.8, min_images=10):
    """
    Performs iterative calibration with outlier rejection based on reprojection error.
    """
    # Work on copies 
    objpoints = list(objpoints)
    imgpoints = list(imgpoints)
    files = list(files)

    while True:
        # 1) Calibrate using current set
        ret, K, d, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_shape, None, None,
            flags=cv2.CALIB_ZERO_TANGENT_DIST
        )

        # 2) Compute per-image reprojection errors (in pixels)
        errs = per_image_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, d)
        mean_err = float(np.mean(errs))
        max_err = float(np.max(errs))
        worst_i = int(np.argmax(errs))

        print(f"Images used: {len(files)} | RMS(ret): {ret:.4f} | mean px err: {mean_err:.3f} | max px err: {max_err:.3f}")
        print(f"  worst image: {files[worst_i]}  (mean corner error {errs[worst_i]:.3f}px)")

        # 3) Stopping rules
        if mean_err <= max_mean_err:
            break
        if len(files) <= min_images:
            print("Reached minimum number of images; stopping.")
            break

        # 4) Remove the worst image and repeat
        print("  removing worst image and recalibrating...\n")
        objpoints.pop(worst_i)
        imgpoints.pop(worst_i)
        files.pop(worst_i)

    return ret, K, d, files

def main():
    objpoints, imgpoints, files, image_shape = detect_corners_all_images("*.jpeg")
    print(f"\nDetected corners in {len(files)} images.\n")

    ret, K, d, kept_files = iterative_reject_calibration(
        objpoints, imgpoints, files, image_shape,
        max_mean_err=0.8,   # quality target (pixels)
        min_images=10       # don't go below this
    )

    np.savez("calibration_results_iterative.npz",
             cameraMatrix=K, distCoeffs=d, rms=ret, kept_files=np.array(kept_files))

    print("\nFinal results saved to calibration_results_iterative.npz")
    print("Final RMS(ret):", ret)
    print("Final K:\n", K)
    print("Final d:\n", d.ravel())
    print("Kept images:", kept_files)

if __name__ == "__main__":
    main()
