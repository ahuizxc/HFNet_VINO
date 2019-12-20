import numpy as np
import cv2
from sklearn.decomposition import PCA


def draw_keypoints(img, kpts, color=(0, 255, 0), radius=4, s=3):
    img = np.uint8(img)
    if s != 1:
        img = cv2.resize(img, None, fx=s, fy=s)
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, -1)
    for k in np.int32(kpts):
        cv2.circle(img, tuple(s*k), radius, color,
                   thickness=-1, lineType=cv2.LINE_AA)
    return img


def draw_matches(img1, kp1, img2, kp2, matches, color=None, kp_radius=5,
                 thickness=2, margin=20):
    # Create frame
    if len(img1.shape) == 2:
        img1 = img1[..., np.newaxis]
    if len(img2.shape) == 2:
        img2 = img2[..., np.newaxis]
    if img1.shape[-1] == 1:
        img1 = np.repeat(img1, 3, -1)
    if img2.shape[-1] == 1:
        img2 = np.repeat(img2, 3, -1)
    new_shape = (max(img1.shape[0], img2.shape[0]),
                 img1.shape[1]+img2.shape[1]+margin,
                 img1.shape[2])
    new_img = np.ones(new_shape, type(img1.flat[0]))*255

    # Place original images
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],
            img1.shape[1]+margin:img1.shape[1]+img2.shape[1]+margin] = img2

    # Draw lines between matches
    if not isinstance(color, list):
        color = [color]*len(matches)
    for m, c in zip(matches, color):
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not c:
            if len(img1.shape) == 3:
                c = np.random.randint(0, 256, 3)
            else:
                c = np.random.randint(0, 256)
            c = (int(c[0]), int(c[1]), int(c[2]))

        end1 = tuple(np.round(kp1[m[0]]).astype(int))
        end2 = tuple(np.round(kp2[m[1]]).astype(int)
                     + np.array([img1.shape[1]+margin, 0]))
        cv2.line(new_img, end1, end2, c, thickness, lineType=cv2.LINE_AA)
        cv2.circle(
            new_img, end1, kp_radius, c, thickness, lineType=cv2.LINE_AA)
        cv2.circle(
            new_img, end2, kp_radius, c, thickness, lineType=cv2.LINE_AA)
    return new_img


def draw_dense_descriptors(*maps):
    pca = PCA(n_components=3, svd_solver='full')
    dims = [m.shape[-1] for m in maps]
    assert len(np.unique(dims)) == 1
    dim = dims[0]
    all_maps = np.concatenate([m.reshape(-1, dim) for m in maps])
    pca.fit(all_maps)
    projected = [pca.transform(m.reshape(-1, dim)).reshape(m.shape[:2]+(3,))
                 for m in maps]
    _min = np.min([np.min(p) for p in projected])
    _max = np.max([np.max(p) for p in projected])
    return [(p - _min) / (_max - _min) for p in projected]
