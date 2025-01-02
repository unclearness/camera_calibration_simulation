import cv2
import numpy as np


def apply_distortion(
    img, fx, fy, cx, cy, width, height, k1, k2, p1, p2, k3=0.0, k4=0.0, k5=0.0, k6=0.0
):
    # Compute scale factor
    # Note that distortion is applied to bigger image to avoid decreasing image quality
    scale = 1.0# max(img.shape[1] / width, 2.0)

    print(scale)

    # Compute scale factor
    fx_org, fy_org, cx_org, cy_org = fx, fy, cx, cy
    fx = fx_org * scale
    fy = fy_org * scale
    cx = cx_org * scale
    cy = cy_org * scale
    K_scaled = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    print(K_scaled)

    h, w = img.shape[:2]

    # Generated grid points
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    print(x.shape, y.shape)
   
    # Apply distortion to grid points
    u1 = (x - cx) / fx
    v1 = (y - cy) / fy
    u2 = u1 * u1
    v2 = v1 * v1
    r2 = u2 + v2
    _2uv = 2 * u1 * v1
    kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) * r2)
    x_distorted = fx * (u1 * kr + p1 * _2uv + p2 * (r2 + 2 * u2)) + cx
    y_distorted = fy * (v1 * kr + p1 * (r2 + 2 * v2) + p2 * _2uv) + cy

    points_distorted = np.stack([x_distorted, y_distorted], axis=-1)
    if False:
        # points_distorted = points_distorted.reshape(-1, 1, 2)
        # points_distorted = cv2.convertPointsToHomogeneous(points_distorted)
        # points_distorted = (K_scaled @ points_distorted.transpose(0, 2, 1)).transpose(
        #     0, 2, 1
        # )
        #print(points_distorted.shape, np.ones((h, w, 1)).shape)
        points_distorted = np.concatenate([points_distorted, np.ones((h, w, 1))], axis=-1)
        #print(K_scaled.shape, points_distorted.shape)
        #print(K_scaled[:2])
        K_scaled_inv = np.linalg.inv(K_scaled)
        print(K_scaled_inv)
        points_distorted = (K_scaled_inv @ points_distorted.reshape(-1, 3).T).reshape(h, w, 3)
        #points_distorted = (K_scaled @ points_distorted.transpose(0, 2, 1)).transpose(
        #    0, 2, 1
        #)
        #print(points_distorted.)
        
        #points_distorted = points_distorted[:, :, :2] / points_distorted[:, :, 2:]
        print(points_distorted)
        print(points_distorted.shape)
        hoge
        # dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
        # pts = np.stack([x, y], axis=-1).astype(np.float32)
        # print(pts.shape)
        # points_distorted = pts #cv2.undistortPoints(pts.reshape(-1, 2), K_scaled, dist_coeffs).reshape(h, w, 2)


        # Map original grid points to distorted points
        #map_x = points_distorted[:, :,  0].reshape(h, w).astype(np.float32) 
        #map_y = points_distorted[:, :,  1].reshape(h, w).astype(np.float32)

    map_x = points_distorted[:, :, 0].astype(np.float32)
    map_y = points_distorted[:, :, 1].astype(np.float32)
 
    #print(map_x, map_y)
    # Apply distortion to image
    distorted = cv2.remap(
        img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
    )

    # Resize (shrink) to original size
    final_image = cv2.resize(
        distorted, (width, height), interpolation=cv2.INTER_LANCZOS4
    )

    return final_image


if __name__ == "__main__":
    pass
