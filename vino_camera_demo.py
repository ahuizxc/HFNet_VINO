import tensorflow as tf
import sys
import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin,IECore
import PIL
import time
import pdb
from hfnet_msgs.msg import Hfnet
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
tf.enable_eager_execution()
frame_data = None

class VinoNetwork:
    def __init__(self, cpu_extension):
        self.ie = IECore()
        self.ie.add_extension(cpu_extension, "CPU")
        self.scaling_desc = (np.array([28,28]) - 1.)/(np.array([224,224]) - 1.)
    def load_vino_net(self, model_file, weights_file):
        self.net = IENetwork(model=model_file, weights=weights_file)
    def build_input_output(self):
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1
    def build_exec_net(self):
        self.exec_net = self.ie.load_network(network=self.net, device_name="CPU")
    def infer(self, image):
        self.image_raw = image
        
        self.scale = [self.image_raw.shape[0] /224, self.image_raw.shape[1] /224]
        image = cv2.resize(image, (224,224))
        self.image = image
        image = image.transpose((2, 0, 1))
        self.res = self.exec_net.infer(inputs={self.input_blob: np.expand_dims(image, axis=0)})
        self.result = list(self.res.values())
    def get_keypoints(self):
        scores = self.res['pred/local_head/detector/Squeeze']
        self.keypoints = tf.where(tf.greater_equal(scores[0], 0.015))
    def draw_keypoints(self):
        self.keypoints = np.array([[int(i[0]*self.scale[1]),int(i[1]*self.scale[0])] for i in self.keypoints.numpy()[..., ::-1]])
        [cv2.circle(self.image_raw, (int(i[0]),int(i[1])), int(1), (255, 0, 0), 2) for i in self.keypoints]
    def get_local_desc(self):
        self.local_descriptors = np.transpose(self.res['pred/local_head/descriptor/Conv_1/BiasAdd/Normalize'],(0,2,3,1))
        self.local_descriptors = \
                    tf.nn.l2_normalize(
                        tf.contrib.resampler.resampler(
                            self.local_descriptors, 
                            tf.to_float(self.scaling_desc)[::-1]*tf.to_float(self.keypoints[None])), -1).numpy()[:50]
    def get_global_desc(self):
        self.global_descriptors = self.res['pred/global_head/dimensionality_reduction/BiasAdd/Normalize']

    def to_ros_msg(self, header):
        msg = Hfnet()
        msg.header = header
        points = []
        local_desc = []

        kp_list = self.keypoints.tolist()
        desc_list = self.local_descriptors[0].tolist()
        for i,kp in enumerate(kp_list):
            p = Point()
            desc = Float32MultiArray()
            desc.data = desc_list[i]
            p.x = kp[0]
            p.y = kp[1]
            local_desc.append(desc)
            points.append(p)
        global_desc = Float32MultiArray()
        global_desc.data = self.global_descriptors[0].tolist()
        msg.local_desc = local_desc
        msg.global_desc = global_desc
        msg.local_points = points
        return msg

if __name__ == "__main__":
    cpu_extension = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx2.so"
    model_file = "./saved_model.xml"
    weights_file = "saved_model.bin"
    vino_net = VinoNetwork(cpu_extension)
    vino_net.load_vino_net(model_file, weights_file)
    vino_net.build_input_output()
    vino_net.build_exec_net()
    capture = cv2.VideoCapture(0)

    # for i in range(90):
    #     _, frame = capture.read()      
    plot_matcher = True
    image_old = None
    keypoints_old = None
    keypoints_desc_old = None
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    frame_count = 0

    while True:
        start_time = time.time()
        _, frame = capture.read()
        vino_net.infer(frame)
        vino_net.get_keypoints()
        vino_net.get_local_desc()
        vino_net.draw_keypoints()
        vino_net.get_global_desc()
        msg = vino_net.to_ros_msg(Header())
        # end_time = time.time()
        if frame_count>0:
            try:
                # pdb.set_trace()
                matches = bf.match(vino_net.local_descriptors[0], keypoints_desc_old[0])
                # pdb.set_trace()
                kp1 = cv2.KeyPoint_convert(vino_net.keypoints[:,None,:])
                kp2 =  cv2.KeyPoint_convert(keypoints_old[:,None,:])
                # pdb.set_trace()
                img = cv2.drawMatches(vino_net.image_raw, 
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
        keypoints_desc_old = vino_net.local_descriptors
        image_old = frame
        keypoints_old = vino_net.keypoints
        key = cv2.waitKey(50)
        if key  == ord('q'):  
            # pdb.set_trace()
            break

        if frame_count>0:
            cv2.putText(img, 
                        "FPS : {:.2f}".format(1/(end_time - start_time)), 
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("video", img)
        frame_count+=1
    cv2.destroyAllWindows()