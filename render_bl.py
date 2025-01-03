import random
import sys
from pathlib import Path
import json

import numpy as np
import bpy
import mathutils
from mathutils import Vector, Matrix, Euler

sys.path.append(str(Path(__file__).parent))
from camera_util_bl import set_intrinsics_from_K_matrix, get_intrinsics_as_K_matrix


def calculate_camera_distance(real_size, fx, target_pixel_size):
    z = (fx * real_size) / target_pixel_size
    return z


def compute_view_frustum(fx, fy, cx, cy, width, height, near, far):
    """
    Compute the 8 vertices of the view frustum from pixel-based intrinsic parameters,
    image width/height, and near/far planes.

    Parameters:
    - fx: Focal length in x direction (pixels)
    - fy: Focal length in y direction (pixels)
    - cx: Principal point x-coordinate (pixels)
    - cy: Principal point y-coordinate (pixels)
    - width: Image width (pixels)
    - height: Image height (pixels)
    - near: Near plane distance
    - far: Far plane distance

    Returns:
    - vertices: 8x3 array where each row is a vertex of the frustum in 3D space.
    """
    # Normalized device coordinates (NDC) [-1, 1] to pixel space
    near_left = -cx / fx * near
    near_right = (width - cx) / fx * near
    near_top = -cy / fy * near
    near_bottom = (height - cy) / fy * near

    far_left = -cx / fx * far
    far_right = (width - cx) / fx * far
    far_top = -cy / fy * far
    far_bottom = (height - cy) / fy * far

    # Vertices: [near bottom-left, near bottom-right, near top-right, near top-left,
    #            far bottom-left, far bottom-right, far top-right, far top-left]
    vertices = np.array(
        [
            [near_left, near_bottom, -near],  # Near bottom-left
            [near_right, near_bottom, -near],  # Near bottom-right
            [near_right, near_top, -near],  # Near top-right
            [near_left, near_top, -near],  # Near top-left
            [far_left, far_bottom, -far],  # Far bottom-left
            [far_right, far_bottom, -far],  # Far bottom-right
            [far_right, far_top, -far],  # Far top-right
            [far_left, far_top, -far],  # Far top-left
        ]
    )

    return vertices


def setup_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100


# Set Image Texture as output material
def set_image_texture_material(obj):
    for mat_slot in obj.material_slots:
        mat = mat_slot.material
        if mat and mat.use_nodes:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Find Image Texture Node
            texture_node = None
            for node in nodes:
                if isinstance(node, bpy.types.ShaderNodeTexImage):
                    texture_node = node
                    break

            if texture_node:
                # Clear existing links to Material Output
                for link in links:
                    if link.to_node.type == "OUTPUT_MATERIAL":
                        links.remove(link)

                # Connect Image Texture directly to Material Output
                output_node = [node for node in nodes if node.type == "OUTPUT_MATERIAL"]
                if output_node:
                    links.new(texture_node.outputs[0], output_node[0].inputs[0])
                break


def import_calibration_board(board_obj_path):
    board_obj_path = str(Path(board_obj_path).absolute())

    # Import the OBJ file with original orientation
    bpy.ops.wm.obj_import(filepath=board_obj_path, forward_axis="Y", up_axis="Z")
    board = bpy.context.selected_objects[0]

    set_image_texture_material(board)

    # Disable "Filmic"
    bpy.data.scenes["Scene"].view_settings.view_transform = "Raw"  # "Standard"

    return board


def render_animation(camera, output_dir):
    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(exist_ok=True)

    bpy.context.scene.render.image_settings.file_format = "PNG"
    #bpy.context.scene.render.image_settings.quality = 90

    camera_parameters = []

    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)

        output_path = str(output_dir / f"frame_{frame:04d}_nodist.png")
        bpy.context.scene.render.filepath = output_path

        print(f"Rendering frame {frame}...")
        bpy.ops.render.render(write_still=True)

        cam2world_gl = camera.matrix_world
        cam2world_cv = np.array(cam2world_gl.copy()) 

        # Flip y and z axes to align with OpenCV coordinate system
        cam2world_cv[..., 1] = -cam2world_cv[..., 1]
        cam2world_cv[..., 2] = -cam2world_cv[..., 2]

        K = get_intrinsics_as_K_matrix()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # Save camera parameters
        camera_parameters.append(
            {
                "view": frame,
                "image": Path(output_path).name,
                "intrinsic": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "cam2world_cv": [list(row) for row in cam2world_cv],
            }
        )

    # Save camera parameters to JSON file
    camera_params_path = str(Path(output_dir) / "camera_parameters.json")
    with open(camera_params_path, "w") as json_file:
        json.dump(camera_parameters, json_file, indent=4)

    print("Rendering completed!")


def look_at(obj, target, up):
    R = mathutils.Matrix.Identity(3)
    direction = obj.location - target
    direction.normalize()
    R[0][2] = direction.x
    R[1][2] = direction.y
    R[2][2] = direction.z

    right = up.cross(direction).normalized()
    R[0][0] = right.x
    R[1][0] = right.y
    R[2][0] = right.z

    new_up = right.cross(direction).normalized()
    R[0][1] = new_up.x
    R[1][1] = new_up.y
    R[2][1] = new_up.z

    obj.rotation_euler = R.to_euler()


def create_camera_calibration_animation(
    camera,
    size,
    cam_z,
    view_frustum,
    obj,
    rows=10,
    cols=5,
    trans_noise_stddev=0.0,
    rot_noise_stddev=0.0,
):
    num_frames = rows * cols
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames - 1

    camera.location = Vector((0, 0, 0))
    camera.rotation_euler = Euler((0, 0, 0))

    offset = max(size) * 0.25
    left = view_frustum[4][0] + size[0] / 2 + offset
    right = view_frustum[5][0] - size[0] / 2 - offset
    top = view_frustum[6][1] + size[1] / 2 + offset
    bottom = view_frustum[5][1] - size[1] / 2 - offset

    rot_noise_stddev = np.pi / 180 * 5

    center = (left + right) / 2, (top + bottom) / 2, -cam_z

   # Accumulate vertex coordinates
    centroid = mathutils.Vector((0.0, 0.0, 0.0))
    for vert in obj.data.vertices:
        centroid += vert.co
    
    # Calculate the average
    centroid /= len(obj.data.vertices)
    
    # Convert to world coordinates
    world_centroid = obj.matrix_world @ centroid

    random.seed(0)

    for row in range(rows):
        x = left + (right - left) * row / (rows - 1)
        for col in range(cols):
            y = bottom + (top - bottom) * col / (cols - 1)
            z = - cam_z
            x += random.uniform(-trans_noise_stddev, trans_noise_stddev)
            y += random.uniform(-trans_noise_stddev, trans_noise_stddev)
            z += random.uniform(-trans_noise_stddev, trans_noise_stddev)

            frame = row * cols + col
            bpy.context.scene.frame_set(frame)

            base_rx = np.pi
            rx = random.uniform(-rot_noise_stddev*10, rot_noise_stddev*10)
            ry = random.uniform(-rot_noise_stddev*5, rot_noise_stddev*5)
            rz = random.uniform(-rot_noise_stddev, rot_noise_stddev)
            # import math
            # rx = math.radians(45)
            # ry = 0
            # rz = 0

            if False:
                camera.location = Vector((x, y, z))
                camera.rotation_euler = Euler((rx + base_rx, ry, rz))
            else:
                #r = np.abs(cam_z)
                camera.location = Vector((x, y, z))
                #print("org", camera.location)

                R = np.array(Euler((rx, ry, rz)).to_matrix())
                
                new_camera_rotation = R.T @ np.array(Euler((base_rx, 0, 0)).to_matrix())
                
                new_camera_pos = np.dot(R.T, (np.array(camera.location) - np.array(world_centroid))) + np.array(world_centroid)

                camera.location = Vector(new_camera_pos)
                rx, ry, rz = Matrix(new_camera_rotation).to_euler()
                camera.rotation_euler = Euler((rx, ry, rz))
                #print("new", camera.location, camera.rotation_euler)
                #print()

            camera.keyframe_insert(data_path="location")
            camera.keyframe_insert(data_path="rotation_euler")

def main(board_obj_path, intrin_json_path, output_dir):

    setup_scene()

    # Setup camera intrinsics EXCEPT distortion
    with open(intrin_json_path, "r") as json_file:
        intrin = json.load(json_file)
        fx = float(intrin["fx"])
        fy = float(intrin["fy"])
        cx = float(intrin["cx"])
        cy = float(intrin["cy"])
        width = int(intrin["width"])
        height = int(intrin["height"])

    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera

    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    set_intrinsics_from_K_matrix(K, width, height)

    # Setup calibration board
    obj = import_calibration_board(board_obj_path)

    bbox_corners = [
        obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box
    ]
    min_corner = mathutils.Vector(
        (
            min(c[0] for c in bbox_corners),
            min(c[1] for c in bbox_corners),
            min(c[2] for c in bbox_corners),
        )
    )
    max_corner = mathutils.Vector(
        (
            max(c[0] for c in bbox_corners),
            max(c[1] for c in bbox_corners),
            max(c[2] for c in bbox_corners),
        )
    )
    center = (min_corner + max_corner) / 2
    size = max_corner - min_corner

    # Calculate camera distance
    rough_rows = 3
    cam_z = calculate_camera_distance(size[1], fy, height / rough_rows)

    # Create camera pose animation
    view_frustum = compute_view_frustum(fx, fy, cx, cy, width, height, 0.01, cam_z)
    view_frustum[..., 0] += center[0]
    view_frustum[..., 1] += center[1]

    create_camera_calibration_animation(camera, size, cam_z, view_frustum, obj)

    render_animation(camera, output_dir)

    bpy.ops.wm.save_as_mainfile(filepath=str(Path(output_dir) / "scene.blend"))


if __name__ == "__main__":
    # Get arguments passed from the command line
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :]  # Arguments after '--'

    if len(argv) < 2:
        print("Usage: blender -b -P blender_render.py -- <obj_file_path> <output_dir>")
        sys.exit(1)

    board_obj_path = argv[0]
    intrin_json_path = argv[1]
    output_dir = argv[2]

    main(board_obj_path, intrin_json_path, output_dir)
