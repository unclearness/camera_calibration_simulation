import cv2
from pathlib import Path
import numpy as np
import json

def compute_tex_size(img_size, marker_num, marker_len):
    marker_per_pix_x = img_size[0] / marker_num[0]
    marker_per_pix_y = img_size[1] / marker_num[1]

    if marker_per_pix_x < marker_per_pix_y:
        base_len = marker_num[0] * marker_len
        return base_len, base_len * img_size[1] / img_size[0]
    else:
        base_len = marker_num[1] * marker_len
        return base_len * img_size[0] / img_size[1], base_len


# def compute_marker_len_pix(img_size, marker_num):
#     marker_per_pix_x = img_size[0] / marker_num[0]
#     marker_per_pix_y = img_size[1] / marker_num[1]

#     if marker_per_pix_x < marker_per_pix_y:
#         marker_len_pix = img_size[0] / marker_num[0]
#     else:
#         marker_len_pix = img_size[1] / marker_num[1]
#     return marker_len_pix


def compute_marker_margin_pix(img_size, marker_num):
    marker_per_pix_x = img_size[0] / marker_num[0]
    marker_per_pix_y = img_size[1] / marker_num[1]

    if marker_per_pix_x < marker_per_pix_y:
        marker_len_pix = img_size[0] / marker_num[0]
        margin_x = 0.0
        margin_y = (img_size[1] - marker_num[1] * marker_len_pix) / 2
    else:
        marker_len_pix = img_size[1] / marker_num[1]
        margin_x = (img_size[0] - marker_num[0] * marker_len_pix) / 2
        margin_y = 0.0
    return marker_len_pix, margin_x, margin_y


# def make_board_mesh(w, h):
#     vertices = [
#         [-w / 2, -h / 2, 0],
#         [w / 2, -h / 2, 0],
#         [w / 2, h / 2, 0],
#         [-w / 2, h / 2, 0],
#     ]
#     uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
#     faces = [[0, 1, 2], [0, 2, 3]]
#     return vertices, uvs, faces


def make_board_mesh(w, h):
    vertices = [
        [0, 0, 0],
        [w, 0, 0],
        [w, h, 0],
        [0, h, 0],
    ]
    uvs = [[0, 1], [1, 1], [1, 0], [0, 0]]
    faces = [[0, 2, 1], [0, 3, 2]]
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


def write_points(obj_path, points, color=(0, 0, 0)):
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    c = color
    with open(obj_path, "w") as fp:
        for p in points:
            fp.write("v {} {} {} {} {} {}\n".format(p[0], p[1], p[2], c[0], c[1], c[2]))


# Example usage
if __name__ == "__main__":
    # Create a sample ArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    square_len = 12.0
    marker_len = 9.0
    size = (11, 8)
    board = cv2.aruco.CharucoBoard(size, square_len, marker_len, aruco_dict)

    with open("./data/charuco.json", "w") as fp:
        json.dump(
            {
                "dict": "DICT_4X4_100",
                "square_len": square_len,
                "marker_len": marker_len,
                "size": size,
            },
            fp,
        )

    # img = board.generateImage((int(square_len*11), int(square_len*8)), marginSize=0, borderBits=1)

    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for pt in board.getChessboardCorners():
    #     pt = np.round(pt)
    #     print(pt)
    #     pt = (int(pt[0]), int(pt[1]))
    #     cv2.circle(img, pt, 2, (0, 0, 255), -1)

    # for pt in board.getObjPoints():
    #     pt = np.mean(pt, axis=0)
    #     pt = (int(pt[0]), int(pt[1]))
    #     cv2.circle(img, pt, 2, (0, 255, 0), -1)

    # cv2.imwrite("./data/charuco_test.png", img)

    # hoge

    scale = 100.0
    size_img = (int(size[0] * scale), int(size[1] * scale))
    img = board.generateImage(size_img, marginSize=0, borderBits=1)

    # print(len(board.getObjPoints()), len(board.getIds()))
    # print(board.getObjPoints())
    # hoge

    cv2.imwrite("./data/charuco.png", img)

    marker_size_meter = 0.012
    tex_size_meter = compute_tex_size(
        img.shape[:2][::-1], board.getChessboardSize(), marker_size_meter
    )

    pix2meter = tex_size_meter[0] / img.shape[1]

    vertices, uvs, faces = make_board_mesh(tex_size_meter[0], tex_size_meter[1])
    write_obj(Path("./data/charuco.obj"), "charuco.png", vertices, uvs, faces, img)

    org_corners = board.getChessboardCorners()
    # org_marker_size_pix = compute_marker_len_pix((11*12, ), board.getChessboardSize())
    marker_len_pix, margin_x_pix, margin_y_pix = compute_marker_margin_pix(
        img.shape[:2][::-1], board.getChessboardSize()
    )
    corners_mm = org_corners / square_len * marker_size_meter

    margin_x_mm = margin_x_pix * pix2meter
    margin_y_mm = margin_y_pix * pix2meter
    tex_size_meter = list(tex_size_meter)
    tex_size_meter[0] -= margin_x_mm * 2
    tex_size_meter[1] -= margin_y_mm * 2

    # corners_mm[..., 0] -= tex_size_meter[0] / 2
    # corners_mm[..., 1] -= tex_size_meter[1] / 2
    # print(marker_size_pix, marker_size_meter)
    write_points(Path("./data/charuco_corners.obj"), corners_mm, color=(0, 0, 1))

    org_markers = np.array(board.getObjPoints())
    # print(org_markers.shape)
    # Flip Y and Z axes
    # org_markers[:, :, 1] *= -1
    # org_markers[:, :, 2] *= -1

    markers_mm = org_markers / square_len * marker_size_meter
    # markers_mm[..., 0] -= tex_size_meter[0] / 2
    # markers_mm[..., 1] -= tex_size_meter[1] / 2
    markers_mm_flat = markers_mm.reshape(-1, 3)
    write_points(
        Path("./data/charuco_marker_edges.obj"), markers_mm_flat, color=(0, 1, 0)
    )
    write_points(
        Path("./data/charuco_marker_centers.obj"),
        markers_mm.reshape(-1, 4, 3).mean(axis=1),
        color=(1, 0, 0),
    )
