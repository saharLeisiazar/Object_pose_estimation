#!/usr/bin/env python
import rospy

from darknet_ros_msgs.msg import BoundingBoxes
from geometry_msgs.msg import PoseStamped
import message_filters
from sensor_msgs.msg import JointState,Image
from feature_extractor.msg import features
import cv2
import numpy as np
from nav_msgs.msg import Odometry
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import copy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from scipy.spatial.transform import Rotation as RR
import sys
import ast
#######################################

frames =[]
PTS=[]
DES=[]
PTS1=[]

class object_pose_detector():

    def __init__(self):      
        self.prev_cam=np.array([0,0,0])
        self.xmidd_prev = 0
        self.ymidd_prev = 0
        self.prev_objects = []
        self.img_counter = 0
        rospy.init_node('listener', anonymous=True)

        publisher = rospy.Publisher("points3D", MarkerArray, queue_size=10)

        I1 = cv2.imread(input_image_1)
        I2 = cv2.imread(input_image_2)

        img1_xmin = boundingbox_1[0]
        img1_ymin = boundingbox_1[1]
        img1_xmax = boundingbox_1[2]
        img1_ymax = boundingbox_1[3]

        img2_xmin = boundingbox_2[0]     
        img2_ymin = boundingbox_2[1]
        img2_xmax = boundingbox_2[2]
        img2_ymax = boundingbox_2[3]

        z1 = 0.0   # x odom
        x1 = -1 * 0.0 # -y odom
        y1 = -1 * 0  # -z odom
        r1 = 0
        p1 = 0
        ya1 = 0.0
        w1 = 1

        z2 = pose_diff[0]   #0.0
        x2 = -1 * pose_diff[1]   #-0.12
        y2 = -1 * pose_diff[2]   #0
        r2 = 0
        p2 = 0
        ya2 = 0.0
        w2 = 1

        c=np.zeros(3)
        R=np.zeros(4)

        c[0] = x1
        c[1] = y1
        c[2] = z1

        R[0] = r1
        R[1] = p1
        R[2] = ya1
        R[3] = w1


        '''
        Projection Matrix
        '''

        K = np.array([[528.96002, 0, 620.22998],[0, 528.66498, 368.64499],[0, 0, 1]])
    
        Rot = np.array([[1-2*(R[1]**2)-2*(R[2]**2),2*R[0]*R[1]+2*R[2]*R[3],2*R[0]*R[2]-2*R[1]*R[3],0],
                        [2*R[0]*R[1]-2*R[2]*R[3],1-2*(R[0]**2)-2*(R[2]**2),2*R[1]*R[2]+2*R[0]*R[3],0],
                        [2*R[0]*R[2]+2*R[1]*R[3],2*R[1]*R[2]-2*R[0]*R[3],1-2*(R[0]**2)-2*(R[1]**2),0],
                        [0,0,0,1]])

        Tran = np.array([[1,0,0,-c[0]],
                        [0,1,0, -c[1]],
                        [0,0,1, -c[2]],
                        [0,0,0,   1]])             
        Proj = np.dot(K,np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]]))
        Proj = np.dot(Proj,Rot)
        Proj = np.dot(Proj,Tran)
        prev_proj = Proj

        c[0] = x2
        c[1] = y2
        c[2] = z2

        R[0] = r2
        R[1] = p2
        R[2] = ya2
        R[3] = w2

        K = np.array([[528.96002, 0, 620.22998],[0, 528.66498, 368.64499],[0, 0, 1]])

        Rotation = RR.from_quat([0,0,ya2,w2])
        euler = Rotation.as_euler('zyx', degrees=False)

        Rot= np.array([[ np.cos(euler[0]),0,np.sin(euler[0]),0],
                        [ 0,  1,  0  ,0],
                        [-np.sin(euler[0]),0,np.cos(euler[0]),0],
                        [0,0,0,1]])

        Tran = np.array([[1,0,0,-c[0]],
                        [0,1,0, -c[1]],
                        [0,0,1, -c[2]],
                        [0,0,0,   1]])             

        Proj = np.dot(K,np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]]))
        Proj = np.dot(Proj,Rot)
        Proj = np.dot(Proj, Tran)
        cur_proj = Proj


        '''
        Feature detection
        '''
        thresh = 30
        img1_xmin_crop = max(img1_xmin-thresh , 1)
        img2_xmin_crop = max(img2_xmin-thresh , 1)
        img1_xmax_crop = min(img1_xmax + thresh , I1.shape[1] - 1)
        img2_xmax_crop = min(img2_xmax + thresh , I2.shape[1] - 1)

        img1_ymin_crop = max(img1_ymin-thresh , 1)
        img2_ymin_crop = max(img2_ymin-thresh , 1)
        img1_ymax_crop = min(img1_ymax + thresh , I1.shape[1] - 1)
        img2_ymax_crop = min(img2_ymax + thresh , I2.shape[1] - 1)

        crop_img_prev = I1[img1_ymin_crop:img1_ymax_crop, img1_xmin_crop:img1_xmax_crop]
        crop_img_cur = I2[img2_ymin_crop:img2_ymax_crop , img2_xmin_crop:img2_xmax_crop] 

        orb = cv2.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(crop_img_prev,None)  #crop_img_prev
        print("number of detected features:")
        print(len(kp))
        PTS.append(kp)
        DES.append(des)     
        kp, des = orb.detectAndCompute(crop_img_cur,None)   #crop_img_cur
        PTS.append(kp)
        DES.append(des)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        matches = bf.match(DES[0],DES[1])
        matches = sorted(matches, key = lambda x:x.distance) 
        print("number of matched features:") 
        print(len(matches))  

        numBestMatches = 10
        prev_gray = cv2.cvtColor(crop_img_prev, cv2.COLOR_BGR2GRAY)  #crop_img_prev
        cur_gray = cv2.cvtColor(crop_img_cur, cv2.COLOR_BGR2GRAY)  #crop_img_cur

        mm = matches[:numBestMatches]
        
        img3 = cv2.drawMatches(prev_gray, PTS[0], cur_gray, PTS[1], matches[:numBestMatches], None, flags=2)
        cv2.imshow('img3', img3)
        cv2.waitKey(0)        


        kp_cur = cv2.KeyPoint_convert(PTS[1])
        kp_prev = cv2.KeyPoint_convert(PTS[0])

        list_kp_cur = []
        list_kp_prev = []

        matches = matches[:numBestMatches]
        for mat in matches:
            img_prev_idx = mat.queryIdx
            img_cur_idx = mat.trainIdx
            (x1, y1) = kp_cur[img_cur_idx] + [img2_xmin_crop, img2_ymin_crop]
            (x2, y2) = kp_prev[img_prev_idx] + [img1_xmin_crop, img1_ymin_crop]

            list_kp_cur.append((x1, y1, 1))
            list_kp_prev.append((x2, y2, 1))

        list_kp_cur = np.array(list_kp_cur)
        list_kp_prev = np.array(list_kp_prev)

        cur_points = np.array(list_kp_cur)
        cur_points = np.transpose(cur_points)

        prev_points = np.array(list_kp_prev)
        prev_points = np.transpose(prev_points)
 
        X = cv2.triangulatePoints( prev_proj[:3], cur_proj[:3], prev_points[:2], cur_points[:2] )
        X /= X[3]

        x1 = np.dot(prev_proj[:3],X)
        x2 = np.dot(cur_proj[:3],X)
        
        x1 /= x1[2]
        x2 /= x2[2]

        err_x1 = x1.astype(int) - prev_points.astype(int)
        err_x2 = x2.astype(int) - cur_points.astype(int)

        error_thr = 5

        for j in range(x1.shape[0]-1, -1, -1):
                if err_x1[0][j] >error_thr or err_x1[1][j]>error_thr or err_x2[0][j]>error_thr or err_x2[1][j]>error_thr:
                        x1 = np.delete(x1, j, 1)
                        x2 = np.delete(x2, j, 1)
                        X = np.delete(X, j, 1)
                        err_x1 = np.delete(err_x1, j, 1)
                        err_x2 = np.delete(err_x2, j, 1)


        markerArray = MarkerArray()
        X = np.transpose(X)

        print("3d points")
        print(X)
        print("The error between first images and its projected image. (it should be almost zero for each pixel)")
        print(np.transpose(err_x1))
        print("The error between second images and its projected image. (it should be almost zero for each pixel)")
        print(np.transpose(err_x2))



if __name__ == '__main__':   

     input_image_1 = str(sys.argv[1])  
     input_image_2 = str(sys.argv[2]) 
     pose_diff = ast.literal_eval(sys.argv[3])

     if len(pose_diff) != 3:
        print("pose_diff should be [x, y, z]")

     boundingbox_1 = ast.literal_eval(sys.argv[4])
     boundingbox_2 = ast.literal_eval(sys.argv[5])

     if len(boundingbox_1) != 4 or len(boundingbox_2) != 4:
        print("pose_diff should be [xmin, ymin, xmax, ymax]")

     object_pose_detector()
     rospy.spin()
