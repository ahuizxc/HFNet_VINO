import tensorflow as tf
import sys
import os
import cv2
import numpy as np
from tensorflow.python.saved_model import tag_constants
import PIL
import time

class TFNetwork:
    def __init__(self, model_path):
        self.model_path = model_path
        self.graph2 = tf.Graph()
        self.graph2.as_default()
        self.sess  = tf.Session(graph=self.graph2)
        self.input_size = (224,224)
        tf.saved_model.loader.load(
                self.sess,
                [tag_constants.SERVING],
                "./"
            )
    def build_input_output(self):
        self.net_image_in = self.graph2.get_tensor_by_name('image:0')
        self.net_image_in =  self.graph2.get_tensor_by_name('image:0')

        self.net_scores =  self.graph2.get_tensor_by_name('scores:0')
        self.net_logits =  self.graph2.get_tensor_by_name("logits:0")
        self.net_scores_dense =  self.graph2.get_tensor_by_name("scores_dense:0")
        self.net_local_desc =  self.graph2.get_tensor_by_name('local_descriptors:0')
        self.net_global_decs =  self.graph2.get_tensor_by_name("global_descriptor:0")
        self.net_local_desc_map = self.graph2.get_tensor_by_name("local_descriptor_map:0")
        self.keypoints_op = tf.where(tf.greater_equal(self.net_scores[0], 0.015))
        self.scaling_op = ((tf.cast(tf.shape(self.net_local_desc)[1:3], tf.float64) - 1.)
            / (tf.cast(tf.shape(self.net_image_in)[1:3], tf.float64) - 1.))
        self.local_descriptors_op = tf.nn.l2_normalize(tf.contrib.resampler.resampler(
                    self.net_local_desc, tf.to_float(self.scaling_op)[::-1]*tf.to_float(self.keypoints_op[None])), -1)
    def infer(self, image):
        self.image_raw = image
        
        self.scale = [self.image_raw.shape[0] /224, self.image_raw.shape[1] /224]
        image = cv2.resize(image, self.input_size)
        self.image = image
        self.results = self.sess.run([self.net_scores,
                self.net_scores_dense,
                self.net_logits,
                self.net_local_desc,
                self.net_global_decs,
                self.net_local_desc_map,
                self.local_descriptors_op,
                self.keypoints_op],
                feed_dict = {self.net_image_in: image[None]})

    def get_keypoints(self):
        self.keypoints = self.results[-1]
        self.keypoints = np.array([[int(i[0]*self.scale[1]),int(i[1]*self.scale[0])] for i in self.keypoints[..., ::-1]])
    def draw_keypoints(self):
        [cv2.circle(self.image_raw, (int(i[0]),int(i[1])), int(1), (255, 0, 0), 2) for i in self.keypoints]
    def get_local_desc(self):
        self.local_descriptors = self.results[-2][:50]


import pdb

if __name__ == "__main__":
    tf_net = TFNetwork(model_path="./")
    tf_net.build_input_output()
    capture = cv2.VideoCapture(0)

    plot_matcher = True
    image_old = None
    keypoints_old = None
    keypoints_desc_old = None
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    frame_count = 0
    while True:
        start_time = time.time()
        _, frame = capture.read()
        tf_net.infer(frame)
        tf_net.get_keypoints()
        tf_net.draw_keypoints()
        tf_net.get_local_desc()
        if frame_count>0:
            try:
                matches = bf.match(tf_net.local_descriptors[0], keypoints_desc_old[0])
                kp1 = cv2.KeyPoint_convert(tf_net.keypoints[:,None,:])
                kp2 =  cv2.KeyPoint_convert(keypoints_old[:,None,:])
                img = cv2.drawMatches(tf_net.image_raw, 
                                        kp1,
                                        image_old,
                                        kp2,
                                        matches, None, flags=2)
                end_time = time.time()
            except:
                end_time = time.time()
                pass
        else:
            end_time = time.time()
        keypoints_desc_old = tf_net.local_descriptors
        image_old = frame
        keypoints_old = tf_net.keypoints
        key = cv2.waitKey(50)
        if key  == ord('q'):  
            break
        
        if frame_count>0:
            cv2.putText(img, 
                        "FPS : {:.2f}".format(1/(end_time - start_time)), 
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("video", img)
        frame_count+=1
    cv2.destroyAllWindows()