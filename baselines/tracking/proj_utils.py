import cv2
import numpy as np
import math

# from
# https://blender.stackexchange.com/questions/16472/how-can-i-get-the-cameras-projection-matrix


# This cam was extracted from the image_generation/render_videos code, for the
# camera fixed case
CATER_CAM = [
    (1.4503, 1.6376,  0.0000, -0.0251),
    (-1.0346, 0.9163,  2.5685,  0.0095),
    (-0.6606, 0.5850, -0.4748, 10.5666),
    (-0.6592, 0.5839, -0.4738, 10.7452)]


def project_3d_point(pts):
    """
    Args:
        pts: Nx3 matrix, with the 3D coordinates of the points to convert
    Returns:
        Nx2 matrix, with the coordinates of the point in 2D, from +1 to -1 for
            each dimension. The top left corner is the -1, -1
    """
    p = np.matmul(
        np.array(CATER_CAM),
        np.hstack((pts, np.ones((pts.shape[0], 1)))).transpose()).transpose()
    # The predictions are -1 to 1, Negating the 2nd to put low Y axis on top
    p[:, 0] /= p[:, -1]
    p[:, 1] /= -p[:, -1]
    return p[:, :2]


# To go back from the image to the 3D space, lets learn a 3x3 homography
# projection between the CATER plane (z=0.3421497941017151) and the image plane
# P_cater = P_cam * H (where each row is a pt in P) => H = inv(P_cam) * P_cater
# 4 points should be enough to compute a H
Z = 0.3421497941017151
points_3d = np.array([
    [-3, -3, Z],
    [0, 3, Z],
    [-3, 0, Z],
    [0, 0, Z],
])
points_img = project_3d_point(points_3d)
H, status = cv2.findHomography(points_img, points_3d[:, :2])


def get_class_prediction(cx, cy, nrows=3, ncols=3):
    """
    Uses the homography learned above to project the points from 3D
    The cx, cy MUST be in the format returned by project_3d_point function,
        i.e. between -1 to 1.
    """
    pt_3d_plane = cv2.perspectiveTransform(
        np.array([cx, cy]).reshape((-1, 1, 2)), H)
    x = pt_3d_plane[0, 0, 0]
    y = pt_3d_plane[0, 0, 1]
    # Clip to the area we care about, in case the point was projected to
    # outside, lets bring it in. Since it's a classification problem..
    x = min(max(-3, x), 3-0.00001)
    y = min(max(-3, y), 3-0.00001)
    # Project to the nrows/cols. The points are originally labeled w.r.t
    # a 6x6 grid when rendering the video, so need to be decoded.
    x *= ncols / 3.0
    y *= nrows / 3.0
    # Using the formula from the gen_train_test function that generates the
    # labels
    x1, y1 = (int(math.floor(x)) + ncols,
              int(math.floor(y)) + nrows)
    cls_id = y1 * (2 * ncols) + x1
    assert cls_id >= 0 and cls_id < (4 * nrows * ncols), \
        'cls_id: {} x: {} y: {}'.format(cls_id, x, y)
    return cls_id


def main():
    # Debugging
    im = cv2.imread('/home/rgirdhar/Temp/cater_frames/frames000001.jpg')
    h, w, _ = im.shape
    p2 = project_3d_point(np.array([[0, -3, 0.03]]))
    x1 = int((p2[0, 0] + 1) * w / 2)
    y1 = int((p2[0, 1] + 1) * h / 2)
    im = cv2.circle(im, (x1, y1), 5, (0, 0, 255), -1)
    cv2.imwrite('/home/rgirdhar/Temp/cater_frames/marked.jpg', im)


if __name__ == '__main__':
    main()
