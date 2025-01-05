import cv2
import numpy as np
import json
from pathlib import Path

def load_obj_points(obj_path):
    with open(obj_path, "r") as obj_file:
        lines = obj_file.readlines()

    points = []
    for line in lines:
        if line.startswith("v "):
            splitted = line.split()
            x = splitted[1]
            y = splitted[2]
            z = splitted[3]
            points.append([float(x), float(y), float(z)])

    return np.array(points, dtype=np.float32)


def validate(width, height, corner_points_path, imgage_points_path):

    obj_points = load_obj_points(corner_points_path)

    # Load image points
    with open(imgage_points_path, "r") as fp:
        image_points = json.load(fp)

    img_points = [points["charuco_corners"] for points in image_points if len(points["charuco_corners"]) == len(obj_points)]
    # for pt in img_points:
    #     print(len(pt))
    img_points = np.array(img_points, dtype=np.float32)

    #print(obj_points.shape, img_points.shape)

    #camera_matrix = np.array([[4113.766155339797, 0.0, 910.0],[0.0, 4546.794171691355, 590.0], [0.0, 0.0, 1.0]])
    camera_matrix = np.array([[4000, 0.0, 960.0],[0.0, 4000.794171691355, 520.0], [0.0, 0.0, 1.0]])
    #dist_coeffs = np.array()
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    flags = None
    camera_matrix = None
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        np.array([obj_points] * len(img_points)),
        img_points,
        (width, height),
        camera_matrix,
        None,
        flags=flags
    )

    print(ret)
    print(mtx)
    print(dist)
    # print(rvecs)
    # print(tvecs)


def detect(input_dir, output_dir):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    square_len = 12.0
    marker_len = 9.0
    size = (11, 8)
    board = cv2.aruco.CharucoBoard(size, square_len, marker_len, aruco_dict)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parameters = cv2.aruco.DetectorParameters()
    ch_params = cv2.aruco.CharucoParameters()
    ch_detector = cv2.aruco.CharucoDetector(
        board=board,charucoParams=ch_params, 
        detectorParams=parameters, 
        # refineParams=refine_params
    )

    image_points = []
    for img_path in Path(input_dir).glob("*.png"):
        if img_path.name.endswith("_nodist.png"):
            continue
        img = cv2.imread(str(img_path))
        img_vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = ch_detector.detectBoard(gray)
        # marker_corners, marker_ids = cv2.aruco.ArucoDetector.detectMarkers(gray, aruco_dict, parameters=parameters)
        if len(marker_corners) > 0:
            cv2.aruco.drawDetectedMarkers(img_vis, marker_corners, marker_ids)
            # charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board)
            if len(charuco_ids) > 0:
                color = (255, 0, 0)
                cv2.aruco.drawDetectedCornersCharuco(img_vis, charuco_corners, charuco_ids, color)
        output_path = str(output_dir / (img_path.stem + "_detect.jpg"))
        cv2.imwrite(output_path, img_vis)
        charuco_corners = np.squeeze(charuco_corners)
        marker_corners = np.squeeze(marker_corners)
        marker_centers = np.mean(marker_corners, axis=1)
        image_points.append(
            {
                "image": img_path.name,
                "marker_corners": marker_corners.tolist(),
                "charuco_centers": marker_centers.tolist(),
                "charuco_corners": charuco_corners.tolist(),
            }
        )
    with open(output_dir / "image_points.json", "w") as fp:
        json.dump(image_points, fp)


if __name__ == "__main__":
    #detect("render", "render/detected")

    validate(
        1920, 1080,
        "data/charuco_corners.obj",
        "render/detected/image_points.json"
    )
