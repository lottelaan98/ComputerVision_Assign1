import cv2
import glob

CHECKERBOARD_SIZE = (9, 6)   # inner corners (cols, rows)
IMAGE_PATH = "*.jpeg"
SHOW_IMAGES = True

def test_chessboard_detection():
    """
    Tests all images and checks whether OpenCV can automatically
    detect chessboard corners.
    """
    images = sorted(glob.glob(IMAGE_PATH))

    auto_detected = []
    manual_required = []

    if SHOW_IMAGES:
        cv2.namedWindow("Chessboard Detection Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chessboard Detection Test", 1280, 800)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            auto_detected.append(fname)
            cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners, found)
            status = "AUTO DETECTED"
            color = (0, 255, 0)
        else:
            manual_required.append(fname)
            status = "MANUAL REQUIRED"
            color = (0, 0, 255)

        print(f"{fname}: {status}")

        if SHOW_IMAGES:
            cv2.putText(
                img,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )

            cv2.imshow("Chessboard Detection Test", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    print("\n==================== SUMMARY ====================")
    print(f"Total images: {len(images)}")
    print(f"Auto detected: {len(auto_detected)}")
    print(f"Manual required: {len(manual_required)}")

    print("\nImages requiring MANUAL annotation:")
    for f in manual_required:
        print("  ", f)


if __name__ == "__main__":
    test_chessboard_detection()