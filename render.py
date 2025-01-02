import subprocess
import os
import platform
import shutil
import re
import argparse
import json

import cv2
import numpy as np

from postprocess import apply_distortion

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


def main(
    blender_path,
    output_dir,
    obj_path,
    intrin_json_path,
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

    #blender_path = "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe"

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Command to run Blender with the script
    command = [
        blender_path,
        "-b",  # Run in background
        "-P",
        script_path,  # Path to the Blender Python script
        "--",  # Arguments after '--' are passed to the script
        obj_path,
        intrin_json_path,
        output_dir,
    ]

    # Run the command
    #subprocess.run(command)

    # Setup camera intrinsics with distortion
    with open(intrin_json_path, "r") as json_file:
        intrin = json.load(json_file)
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

    # Load rendered image
    for filename in os.listdir(output_dir):
        if filename.endswith("_nodist.jpg"):
            img = cv2.imread(os.path.join(output_dir, filename), -1)

            # Apply distortion
            img_distorted = apply_distortion(
                img, fx, fy, cx, cy, width, height, k1, k2, p1, p2, k3, k4, k5, k6
            )
            # Save distorted image
            filename = filename.replace("_nodist.jpg", ".jpg")
            cv2.imwrite(os.path.join(output_dir, filename), img_distorted)


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
    parser.add_argument("--intrin_json_path", help="Path to json", required=True)
    parser.add_argument("--output_dir", help="Output directory", required=True)

    args = parser.parse_args()
    blender_path = args.blender_path
    output_dir = os.path.abspath(args.output_dir)
    obj_path = os.path.abspath(args.obj_path)
    intrin_json_path = os.path.abspath(args.intrin_json_path)
    script_path = args.script_path

    main(blender_path, output_dir, obj_path, intrin_json_path, script_path)
