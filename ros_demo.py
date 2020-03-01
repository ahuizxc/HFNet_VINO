import tensorflow as tf
import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin,IECore
import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hfnet_msgs.msg import Hfnet
from geometry_msgs.msg import Point
import pdb
from std_msgs.msg import Float32MultiArray, Float32
tf.enable_eager_execution()
frame_data = None

cpu_extension = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so"
model_file = "saved_model.xml"
weights_file = "saved_model.bin"

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
        # pdb.set_trace()
        msg.global_desc = global_desc
        msg.local_desc = local_desc
        msg.local_points = points
        return msg

def main():
    vino_net = VinoNetwork(cpu_extension)
    vino_net.load_vino_net(model_file, weights_file)
    vino_net.build_input_output()
    vino_net.build_exec_net()
    node = Node(vino_net)
    rospy.spin()


class Node():
    def __init__(self, net):
        self.net = net
        rospy.init_node('hfnet')
        input_topic = rospy.get_param('input_topic', '/usb_cam/image_raw')
        output_topic = rospy.get_param('output_topic', '/features')
        self.subscriber = rospy.Subscriber(input_topic, Image, self.callback)
        self.cv_bridge = CvBridge()
        self.publisher = rospy.Publisher(output_topic, Hfnet)
    def callback(self, msg):
        start_time = time.time()
        frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        self.net.infer(frame)
        self.net.get_keypoints()
        self.net.get_local_desc()
        self.net.draw_keypoints()
        self.net.get_global_desc()
        end_time = time.time()
        feature_msg = self.net.to_ros_msg(msg.header)
        self.publisher.publish(feature_msg)
        # TODO publish
        #self.net.local_descriptors
        #self.net.keypoints

if __name__ == "__main__":
    main()