""" Augmentation pipeline by Paul-Edouard Sarlin and Rémi Pautrat (ETH Zurich)
    for the re-implementation of SuperPoint (DeTone et al.).
    Code: github.com/rpautrat/SuperPoint
"""

import tensorflow as tf
import cv2 as cv
import numpy as np

from . import photometric_augmentation as photaug
from .homographies import (
    sample_homography, compute_valid_mask, warp_points, filter_points,
    adapt_homography_to_resizing)


def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p


def photometric_augmentation(data, **config):
    with tf.name_scope('photometric_augmentation'):
        primitives = parse_primitives(
            config['primitives'], photaug.augmentations)
        prim_configs = [config['params'].get(
                             p, {}) for p in primitives]

        indices = tf.range(len(primitives))
        if config['random_order']:
            indices = tf.random_shuffle(indices)

        def step(i, image):
            fn_pairs = []
            for j, (p, c) in enumerate(zip(primitives, prim_configs)):
                fn_pairs.append(
                    (tf.equal(indices[i], j),
                     lambda p=p, c=c: getattr(photaug, p)(image, **c)))
            image = tf.case(fn_pairs)
            return i + 1, image

        _, image = tf.while_loop(
            lambda i, image: tf.less(i, len(primitives)),
            step, [0, data['image']], parallel_iterations=1)

    return {**data, 'image': image}


def homographic_augmentation(data, add_homography=False, **config):
    with tf.name_scope('homographic_augmentation'):
        image_shape = tf.shape(data['image'])[:2]
        homography = sample_homography(image_shape, **config['params'])[0]
        warped_image = tf.contrib.image.transform(
                data['image'], homography, interpolation='BILINEAR')
        valid_mask = compute_valid_mask(image_shape, homography,
                                        config['valid_border_margin'])

    ret = {**data, 'image': warped_image, 'valid_mask': valid_mask}
    if 'keypoints' in data:
        warped_points = warp_points(data['keypoints'], homography)
        warped_points = filter_points(warped_points, image_shape)
        ret['keypoints'] = warped_points
    for k in ['local_descriptor_map', 'dense_scores']:
        if k in data:
            data[k].set_shape([None, None, None])
            rescaled_homography = adapt_homography_to_resizing(
                homography, image_shape, tf.shape(data[k]))
            ret[k] = tf.contrib.image.transform(
                data[k], rescaled_homography, interpolation='BILINEAR')
            ret[k+'_valid_mask'] = compute_valid_mask(
                tf.shape(data[k])[:2], rescaled_homography, 0)
    if add_homography:
        ret['homography'] = homography
    return ret


def add_dummy_valid_mask(data):
    with tf.name_scope('dummy_valid_mask'):
        valid_mask = tf.ones(tf.shape(data['image'])[:2], dtype=tf.int32)
    return {**data, 'valid_mask': valid_mask}


def add_keypoint_map(data):
    with tf.name_scope('add_keypoint_map'):
        image_shape = tf.shape(data['image'])[:2]
        kp = tf.minimum(tf.to_int32(tf.round(data['keypoints'])), image_shape-1)
        kmap = tf.scatter_nd(
                kp, tf.ones([tf.shape(kp)[0]], dtype=tf.int32), image_shape)
    return {**data, 'keypoint_map': kmap}


def downsample(image, coordinates, **config):
    with tf.name_scope('gaussian_blur'):
        k_size = config['blur_size']
        kernel = cv.getGaussianKernel(k_size, 0)[:, 0]
        kernel = np.outer(kernel, kernel).astype(np.float32)
        kernel = tf.reshape(tf.convert_to_tensor(kernel), [k_size]*2+[1, 1])
        pad_size = int(k_size/2)
        image = tf.pad(image, [[pad_size]*2, [pad_size]*2, [0, 0]], 'REFLECT')
        image = tf.expand_dims(image, axis=0)  # add batch dim
        image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], 'VALID')[0]

    with tf.name_scope('downsample'):
        ratio = tf.divide(tf.convert_to_tensor(config['resize']), tf.shape(image)[0:2])
        coordinates = coordinates * tf.cast(ratio, tf.float32)
        image = tf.image.resize_images(image, config['resize'],
                                       method=tf.image.ResizeMethod.BILINEAR)

    return image, coordinates
