import subprocess
import os
import platform
import shutil
import re
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

import cv2
import numpy as np

from postprocess import distort


def find_blender_path():
    system = platform.system()

    if system == "Windows":
        # Search for Blender in common installation paths
        base_path = "C:\\Program Files\\Blender Foundation\\"
        if os.path.isdir(base_path):
            versions = [
                dir_name
                for dir_name in os.listdir(base_path)
                if re.match(r"Blender \d+\.\d+(\.\d+)?", dir_name)
            ]
            if versions:
                # Sort versions and pick the latest one
                versions.sort(
                    key=lambda v: list(map(int, re.findall(r"\d+", v))), reverse=True
                )
                latest_path = os.path.join(base_path, versions[0], "blender.exe")
                if os.path.isfile(latest_path):
                    return latest_path
        # Check other possible paths
        possible_paths = [
            "C:\\Program Files (x86)\\Blender Foundation\\Blender\\blender.exe",
            os.path.expanduser(
                "~\\AppData\\Local\\Microsoft\\WindowsApps\\blender.exe"
            ),
        ]
    elif system == "Linux":
        # Check for Blender in common Linux paths
        possible_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            shutil.which("blender"),
        ]
    else:
        raise EnvironmentError("Unsupported operating system: " + system)

    # Check other possible paths
    for path in possible_paths:
        if path and os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Blender executable not found. Please install Blender or specify the path manually."
    )


def colorize(values, min_value, max_value, colormap="bwr"):
    values = values.astype(float)
    h, w = values.shape[:2]
    if len(values.shape) > 2:
        values = values[..., 0]

    # Ensure values are within the range
    values = np.clip(values, min_value, max_value)

    # Normalize the values to 0-1
    normalized = (values - min_value) / (max_value - min_value)

    # Get the colormap
    cmap = plt.get_cmap(colormap)

    # Map the normalized values to colors
    colors = cmap(normalized.ravel())
    # Return the RGB components (first three elements of the tuple) for each value
    return (colors[:, :3].reshape(h, w, 3) * 255).astype("uint8")


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

    return np.array(points)


def main(
    blender_path,
    output_dir,
    obj_path,
    setting_json_path,
    script_path,
):
    if blender_path is None:
        # Find Blender path
        try:
            blender_path = find_blender_path()
        except FileNotFoundError as e:
            print(e)
            exit(1)
    else:
        blender_path = os.path.abspath(blender_path)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Command to run Blender with the script
    command = [
        blender_path,
        "-b",  # Run in background
        "-P",
        script_path,  # Path to the Blender Python script
        "--",  # Arguments after '--' are passed to the script
        obj_path,
        setting_json_path,
        output_dir,
    ]

    # Run the command
    subprocess.run(command)

    # Setup camera intrinsics with distortion
    with open(setting_json_path, "r") as json_file:
        intrin = json.load(json_file)["intrin"]
        fx = float(intrin["fx"])
        fy = float(intrin["fy"])
        cx = float(intrin["cx"])
        cy = float(intrin["cy"])
        width = int(intrin["width"])
        height = int(intrin["height"])
        k1 = float(intrin.get("k1", 0.0))
        k2 = float(intrin.get("k2", 0.0))
        p1 = float(intrin.get("p1", 0.0))
        p2 = float(intrin.get("p2", 0.0))
        k3 = float(intrin.get("k3", 0.0))
        k4 = float(intrin.get("k4", 0.0))
        k5 = float(intrin.get("k5", 0.0))
        k6 = float(intrin.get("k6", 0.0))

    vis_dir = os.path.join(output_dir, "visualize")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir, exist_ok=True)

    with open(os.path.join(output_dir, "camera_parameters.json"), "r") as json_file:
        camera_params = json.load(json_file)

    corners_path = obj_path.replace(".obj", "_corners.obj")
    centers_path = obj_path.replace(".obj", "_marker_centers.obj")
    edges_path = obj_path.replace(".obj", "_marker_edges.obj")

    corner_points = load_obj_points(corners_path)
    center_points = load_obj_points(centers_path)
    edge_points = load_obj_points(edges_path)
    gt_image_points = []

    # Load rendered image
    for filename in os.listdir(output_dir):
        if filename.endswith("_nodist.png"):
            img = cv2.imread(os.path.join(output_dir, filename))

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

            camera_param = None
            for param in camera_params:
                if param["image"] == filename:
                    camera_param = param
                    break
            cam2world_cv = np.array(camera_param["cam2world_cv"])
            world2cam_cv = np.linalg.inv(cam2world_cv)
            tvec = world2cam_cv[:3, 3]
            rvec, _ = cv2.Rodrigues(world2cam_cv[:3, :3])

            corner_img_points, _ = cv2.projectPoints(
                corner_points, rvec, tvec, K, dist_coeffs
            )
            center_img_points, _ = cv2.projectPoints(
                center_points, rvec, tvec, K, dist_coeffs
            )
            edge_img_points, _ = cv2.projectPoints(
                edge_points, rvec, tvec, K, dist_coeffs
            )

            corner_img_points = corner_img_points.squeeze()
            center_img_points = center_img_points.squeeze()
            edge_img_points = edge_img_points.squeeze()

            # Apply distortion
            img_distorted = distort(img, K, dist_coeffs)

            # Move the original image to visualize directory
            shutil.move(os.path.join(output_dir, filename),
                        os.path.join(vis_dir, filename))

            # Save distorted image
            filename = filename.replace("_nodist.png", ".png")
            cv2.imwrite(os.path.join(output_dir, filename), img_distorted)

            gt_image_points.append(
                {
                    "image": filename,
                    "chessboard_corners": corner_img_points.tolist(),
                    "marker_centers": center_img_points.tolist(),
                    "marker_corners": edge_img_points.reshape(-1, 4, 2).tolist(),
                }
            )

            # Draw points on distorted image
            img_distorted_vis = img_distorted.copy()
            for point in corner_img_points:
                cv2.circle(
                    img_distorted_vis, tuple(point.ravel().astype(int)), 5, (255, 0, 0), -1
                )
            for point in edge_img_points:
                cv2.circle(
                    img_distorted_vis, tuple(point.ravel().astype(int)), 5, (0, 255, 0), -1
                )
            for point in center_img_points:
                cv2.circle(
                    img_distorted_vis, tuple(point.ravel().astype(int)), 5, (0, 0, 255), -1
                )

            cv2.imwrite(os.path.join(vis_dir, filename.replace(".png", "_dist.png")), img_distorted_vis)

            # Undistort the image and save
            img_undistorted_vis = cv2.undistort(img_distorted_vis, K, dist_coeffs)
            cv2.imwrite(os.path.join(vis_dir, filename.replace(".png", "_reundist.png")), img_undistorted_vis)

            img_undistorted = cv2.undistort(img_distorted, K, dist_coeffs)
            diff = img_undistorted.astype(float) - img.astype(float)
            color = colorize(diff, -30, 30)
            cv2.imwrite(
                os.path.join(vis_dir, filename.replace(".png", "_diff.png")), color
            )

            # diff = (img_distorted.astype(float) - img.astype(float))
            # color = colorize(diff, -30, 30)
            # cv2.imwrite(os.path.join(vis_dir, filename.replace(".png", "_distdiff.png")), color)
            #
    with open(os.path.join(output_dir, "image_points.json"), "w") as json_file:
        json.dump(gt_image_points, json_file, indent=4)


if __name__ == "__main__":
    argparse.ArgumentParser(description="Render a 3D model using Blender.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blender_path", help="Path to Blender executable", default=None
    )
    parser.add_argument(
        "--script_path",
        help="Path to Blender Python script",
        default=os.path.abspath(__file__) + "/../render_bl.py",
    )
    parser.add_argument("--obj_path", help="Path to OBJ model file", required=True)
    parser.add_argument("--setting_json_path", help="Path to json", required=True)
    parser.add_argument("--output_dir", help="Output directory", required=True)

    args = parser.parse_args()
    blender_path = args.blender_path
    output_dir = os.path.abspath(args.output_dir)
    obj_path = os.path.abspath(args.obj_path)
    setting_json_path = os.path.abspath(args.setting_json_path)
    script_path = args.script_path

    main(blender_path, output_dir, obj_path, setting_json_path, script_path)
