import cv2
import numpy as np


def initDistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type):
    # TODO:
    # Use m1type

    w, h = size
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
    src = np.stack([u, v], axis=-1).reshape(-1, 2)

    more_iter = True
    if more_iter:
        dst = cv2.undistortPointsIter(
            src.astype(np.float32),
            cameraMatrix.astype(np.float32),
            distCoeffs.astype(np.float32),
            R.astype(np.float32),
            P=newCameraMatrix.astype(np.float32),
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-8),
        )
    else:
        dst = cv2.undistortPoints(
            src.astype(np.float32),
            cameraMatrix.astype(np.float32),
            distCoeffs.astype(np.float32),
            R.astype(np.float32),
            P=newCameraMatrix.astype(np.float32),
        )
    dst = dst.reshape(w, h, 2)
    map1 = dst[:, :, 0].astype(np.float32)
    map2 = dst[:, :, 1].astype(np.float32)
    return map1, map2


def distort(
    src,
    cameraMatrix,
    distCoeffs,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
):
    map1, map2 = initDistortRectifyMap(
        cameraMatrix,
        distCoeffs,
        np.eye(3),
        cameraMatrix,
        src.shape[:2][::-1],
        cv2.CV_32FC1,
    )
    return cv2.remap(
        src, map1.T, map2.T, interpolation=interpolation, borderMode=borderMode
    ).astype(src.dtype)


def apply_distortion(
    img, fx, fy, cx, cy, width, height, k1, k2, p1, p2, k3=0.0, k4=0.0, k5=0.0, k6=0.0
):
    # NOTE: inversed/negative distortion coefficients with OpenCV can work somehow:
    # cv2.undistort(img, K, -dist_coeffs)
    # But undistortion process is not very accurate due to iterative algorithm,
    # so, for foward distortion process, it is better to use closed-form solution like below.

    # Compute scale factor
    # Note that distortion is applied to bigger image to avoid decreasing image quality
    scale = max(img.shape[1] / width, 2.0)

    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # print(scale)

    # Compute scale factor
    fx_org, fy_org, cx_org, cy_org = fx, fy, cx, cy
    fx = fx_org * scale
    fy = fy_org * scale
    cx = cx_org * scale
    cy = cy_org * scale
    K_scaled = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    # print(K_scaled)

    h, w = img.shape[:2]

    # # Generated grid points
    # y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # #print(x.shape, y.shape)

    # # Apply distortion to grid points
    # u1 = (x - cx) / fx
    # v1 = (y - cy) / fy
    # u2 = u1 * u1
    # v2 = v1 * v1
    # r2 = u2 + v2
    # _2uv = 2 * u1 * v1
    # kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) * r2)
    # x_distorted = fx * (u1 * kr + p1 * _2uv + p2 * (r2 + 2 * u2)) + cx
    # y_distorted = fy * (v1 * kr + p1 * (r2 + 2 * v2) + p2 * _2uv) + cy

    # points_distorted = np.stack([x_distorted, y_distorted], axis=-1)
    # print("hoge", points_distorted.shape)
    if True:

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        points_distorted = np.stack([x, y], axis=-1).reshape(-1, 2)
        # print(points_distorted.shape, K_scaled.shape, np.array([k1, k2, p1, p2, k3, k4, k5, k6]).shape)
        # points_distorted = cv2.undistortPoints(
        #     points_distorted.astype(np.float32),
        #     K_scaled.astype(np.float32),
        #     np.array([k1, k2, p1, p2, k3, k4, k5, k6]).astype(np.float32),
        #     P=K_scaled,
        # )

        points_distorted = cv2.undistortPointsIter(
            points_distorted.astype(np.float32),
            K_scaled.astype(np.float32),
            np.array([k1, k2, p1, p2, k3, k4, k5, k6]).astype(np.float32),
            R=np.eye(3),
            P=K_scaled,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-8),
        )
        points_distorted = points_distorted.reshape(h, w, 2)
        # print(points_distorted)

        if False:
            # points_distorted = points_distorted.reshape(-1, 1, 2)
            # points_distorted = cv2.convertPointsToHomogeneous(points_distorted)
            # points_distorted = (K_scaled @ points_distorted.transpose(0, 2, 1)).transpose(
            #     0, 2, 1
            # )
            # print(points_distorted.shape, np.ones((h, w, 1)).shape)
            points_distorted = np.concatenate(
                [points_distorted, np.ones((h, w, 1))], axis=-1
            )
            # print(K_scaled.shape, points_distorted.shape)
            # print(K_scaled[:2])
            K_scaled_inv = np.linalg.inv(K_scaled)
            print(K_scaled_inv)
            points_distorted = (
                K_scaled_inv @ points_distorted.reshape(-1, 3).T
            ).reshape(h, w, 3)
            # points_distorted = (K_scaled @ points_distorted.transpose(0, 2, 1)).transpose(
            #    0, 2, 1
            # )
            # print(points_distorted.)

            # points_distorted = points_distorted[:, :, :2] / points_distorted[:, :, 2:]
            print(points_distorted)
            print(points_distorted.shape)
            hoge
            # dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
            # pts = np.stack([x, y], axis=-1).astype(np.float32)
            # print(pts.shape)
            # points_distorted = pts #cv2.undistortPoints(pts.reshape(-1, 2), K_scaled, dist_coeffs).reshape(h, w, 2)

            # Map original grid points to distorted points
            # map_x = points_distorted[:, :,  0].reshape(h, w).astype(np.float32)
            # map_y = points_distorted[:, :,  1].reshape(h, w).astype(np.float32)

    map_x = points_distorted[:, :, 0].astype(np.float32)
    map_y = points_distorted[:, :, 1].astype(np.float32)

    # print(map_x, map_y)
    # Apply distortion to image
    distorted = cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    # Resize (shrink) to original size
    final_image = cv2.resize(distorted, (width, height), interpolation=cv2.INTER_LINEAR)

    return final_image


if __name__ == "__main__":
    img = np.zeros((240, 360, 3), dtype=np.uint8)
    step = 20
    for j in range(step, img.shape[0], step):
        cv2.line(img, (0, j), (img.shape[1], j), (255, 255, 255), 1)
    for i in range(step, img.shape[1], step):
        cv2.line(img, (i, 0), (i, img.shape[0]), (255, 255, 255), 1)
    cv2.imwrite("checkerboard.png", img)

    img = cv2.imread("lenna.jpg")

    fx = img.shape[1] * 2
    fy = fx
    cx, cy = img.shape[1] / 2, img.shape[0] / 2
    width, height = img.shape[1], img.shape[0]
    cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    k1 = 0.7
    k2 = -0.5
    p1 = 0.01
    p2 = 0.02
    k3 = 0.01
    k4 = 0.01
    k5 = 0.01
    k6 = 0.01

    distCoeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

    distorted1 = distort(img, cameraMatrix, distCoeffs)
    cv2.imwrite("distroted1.png", distorted1)
    
    distorted2 = cv2.undistort(img, cameraMatrix, -distCoeffs)
    cv2.imwrite("distroted2.png", distorted2)

    undistorted1 = cv2.undistort(distorted1, cameraMatrix, distCoeffs)
    cv2.imwrite("undistorted1.png", undistorted1)
    cv2.imwrite("diff1.png", np.abs(img.astype(float) - undistorted1.astype(float)))

    undistorted2 = cv2.undistort(distorted2, cameraMatrix, distCoeffs)
    cv2.imwrite("undistorted2.png", undistorted2)
    cv2.imwrite("diff2.png", np.abs(img.astype(float) - undistorted2.astype(float)))
