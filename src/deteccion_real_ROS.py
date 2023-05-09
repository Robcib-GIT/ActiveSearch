#!/usr/bin/env python
# PROGRAMA PARA LEER EL NODO DE ROS E INFERIR LA RED DE TERMICA
from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import os, time
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable

"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.
"""
__author__ =  'Simon Haller <simon.haller at uibk.ac.at>'
__version__=  '0.1'
__license__ = 'BSD'
# Python libs
import sys, time

# numpy and scipy
import numpy as np

import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE=True
global image_np

""" CODIGO PARA SUSCRIBIRSE A UNA IMAGEN
def callback(ros_data):
    global image_np
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
    cv2.imshow('cv_img', image_np)
    cv2.waitKey(2)


if __name__ == '__main__':
    global image_np
    subscriber = rospy.Subscriber("/cv_camera/image_raw/compressed",CompressedImage, callback, queue_size = 1)
   
    rospy.init_node('image_feature', anonymous=True)
    
    try:
        rospy.spin()
        
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()
"""

"""
class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",CompressedImage)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/cv_camera/image_raw/compressed",CompressedImage, self.callback, queue_size = 1)
        #self.subscriber = rospy.Subscriber("/thermal_image_view/compressed",CompressedImage, self.callback, queue_size = 1)
        if VERBOSE :
            print ("subscribed to /camera/image/compressed")


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print ('received image of type: "%s"' % ros_data.format)

        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        cv2.imshow('cv_img', image_np)
        cv2.waitKey(2)

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)
        
        #self.subscriber.unregister()

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
"""

def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

    

def callback(ros_data):
    global image_np
    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:

    frame = cv2.resize(image_np, (1280, 960), interpolation=cv2.INTER_CUBIC)
    RGBimg=Convertir_RGB(frame)
    imgTensor = transforms.ToTensor()(RGBimg)
    imgTensor, _ = pad_to_square(imgTensor, 0)
    imgTensor = resize(imgTensor, 416)
    imgTensor = imgTensor.unsqueeze(0)
    imgTensor = Variable(imgTensor.type(Tensor))


    with torch.no_grad():
        detections = model(imgTensor)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    for detection in detections:
        if detection is not None:
            detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                box_w = x2 - x1
                box_h = y2 - y1
                color = (255, 0, 0)
                color2 = (0, 255, 0)
                color3 = (64, 224, 208)
                thickness = 1
                #print("Se detecto {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                #frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 5)
                #frame = cv2.rectangle(frame, (int(x1), int(y1 + box_h)), (int(x2), int(y1)), color, thickness)
                x1=math.floor(x1)
                x2=math.floor(x2)
                y1=math.floor(y1)
                y2=math.floor(y2)
                box_h=math.floor(box_h)
            #    print("x1",x1)
            #    print ("x2",x2)
                #print("Se detecto {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color3, 2)
                

                cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, color2, 1)# Nombre de la clase detectada
                cv2.putText(frame, str("      %.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 1) # Certeza de prediccion de la clase
    cv2.imshow('frame', Convertir_BGR(RGBimg))
    #cv2.imshow('frame', image_np)


    #cv2.imshow('cv_img', image_np)
    cv2.waitKey(2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=1,  help="Is the video processed video? 1 = Yes, 0 == no" )
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    
    '''
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)


    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]
    '''

    
    # DESDE AQUI SE ENVIAN LOS DATOS A ROS
    global image_np
    subscriber = rospy.Subscriber("/cv_camera/image_raw",CompressedImage, callback, queue_size = 1)
   
    rospy.init_node('image_feature', anonymous=True)
    
    try:
        rospy.spin()
        
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()
