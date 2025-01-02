import cv2
from pathlib import Path


def compute_tex_size(img_size, marker_num, marker_len):
    marker_per_pix_x = img_size[0] / marker_num[0]
    marker_per_pix_y = img_size[1] / marker_num[1]

    if marker_per_pix_x < marker_per_pix_y:
        base_len = marker_num[0] * marker_len
        return base_len, base_len * img_size[1] / img_size[0]
    else:
        base_len = marker_num[1] * marker_len
        return base_len * img_size[0] / img_size[1], base_len


def make_board_mesh(w, h):
    vertices = [
        [-w / 2, -h / 2, 0],
        [w / 2, -h / 2, 0],
        [w / 2, h / 2, 0],
        [-w / 2, h / 2, 0],
    ]
    uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
    faces = [[0, 1, 2], [0, 2, 3]]
    return vertices, uvs, faces


def write_obj(obj_path, tex_name, vertices, uvs, faces, texture):
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    mtl_path = obj_path.with_suffix(".mtl")
    with open(mtl_path, "w") as fp:
        fp.write("newmtl mat0\n")
        fp.write("map_Kd {}\n".format(tex_name))

    with open(obj_path, "w") as fp:
        fp.write("mtllib {}\n".format(mtl_path.name))
        for v in vertices:
            fp.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for uv in uvs:
            fp.write("vt {} {}\n".format(uv[0], uv[1]))
        fp.write("usemtl mat0\n")
        for f in faces:
            fp.write(
                "f {}/{} {}/{} {}/{}\n".format(
                    f[0] + 1, f[0] + 1, f[1] + 1, f[1] + 1, f[2] + 1, f[2] + 1
                )
            )

    tex_path = obj_path.parent / tex_name
    cv2.imwrite(str(tex_path), texture)


# Example usage
if __name__ == "__main__":
    # Create a sample ArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard((11, 8), 12, 9, aruco_dict)
    img = board.generateImage((1600, 1000), marginSize=0, borderBits=1)

    cv2.imwrite("./data/charuco.png", img)

    marker_size_meter = 0.012
    tex_size_meter = compute_tex_size(
        img.shape[:2][::-1], board.getChessboardSize(), marker_size_meter
    )

    vertices, uvs, faces = make_board_mesh(tex_size_meter[0], tex_size_meter[1])
    write_obj(Path("./data/charuco.obj"), "charuco.png", vertices, uvs, faces, img)
