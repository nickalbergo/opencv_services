#!/usr/bin/env python

# height: 360
# width: 640

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math
import collections
from statistics import mean

rospy.init_node('ros_python_cv', anonymous=True)
pub = rospy.Publisher('ee_center', Point, queue_size=10)

bridge = CvBridge()

face_cascade = cv2.CascadeClassifier("haar_cascade.xml")

tracker = cv2.TrackerCSRT_create()
bbox = (287, 23, 86, 320)
init_tracker = True

depth_map = np.empty(shape=(360,640))
old_gray = np.empty(shape=(360,640))
mask = np.empty(shape=(360,640))


lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# change as needed
QUEUE_SIZE = 10
TOLERANCE = 50
recent_depths = collections.deque(QUEUE_SIZE*[0], QUEUE_SIZE)

def within_tolerance(v):
    # not enough data
    global recent_depths
    if (np.count_nonzero(recent_depths) < QUEUE_SIZE):
        return False
    
    avg = np.mean(recent_depths)
    
    if ((v < avg + TOLERANCE) and (v > avg - TOLERANCE)):
        return True
    
    return False

def image_callback(img_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except:
        rospy.logerr("CvBridge Error")

    # convert to gray scale of each frames
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    global init_tracker, bbox, old_gray, p0, mask

    if (init_tracker):
        bbox = cv2.selectROI(cv_image, False)
        tracker.init(gray, bbox)
        init_tracker = False
        old_gray = gray

        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(cv_image)

        # faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        # for (x,y,w,h) in faces:
        #     # cv2.rectangle(cv_image,(x,y),(x+w,y+h),(255,255,0),2) 
        #     # roi_gray = gray[y:y+h, x:x+w]
        #     # roi_color = cv_image[y:y+h, x:x+w]
        #     bbox = (int(x), int(y), int(w), int(h))
        #     tracker.init(gray, bbox)
        #     init_tracker = False
    else:
        ok, bbox = tracker.update(gray)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(cv_image, p1, p2, (255,0,0), 2, 1)
            center = (int((bbox[0] + bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[1] + bbox[3]) / 2))
            cv2.circle(cv_image, center, radius=2, color=(255, 255, 0), thickness=-1)

            adjusted_depth = math.inf

            # play around with this range/thresholding/etc.
            # TODO: handle out of bounds checking
            for x in range(-40,40):
                for y in range(-40,40):
                    if ((center[1] < 40) or (center[1] > 600) or (center[0] > 320) or (center[0] < 40)):
                        continue
                    current_pixel = depth_map[center[1] + x][center[0] + y]
                    if (not math.isnan(depth_map[center[1] + x][center[0] + y])):
                        if ((current_pixel < adjusted_depth) and (current_pixel != 0)):
                            adjusted_depth = current_pixel

            if ((adjusted_depth != math.inf) and (within_tolerance(adjusted_depth))):
                # print("Depth at " + str(center) + ": " + str(depth_map[center[1]][center[0]]))
                print("Depth at " + str(center) + ": " + str(adjusted_depth))
                #print("Average: " + str(mean(recent_depths)))
                c = Point(center[0], center[1], adjusted_depth)
                pub.publish(c)

            recent_depths.append(adjusted_depth)     

            p0 = np.empty([1, 1, 2], dtype=np.float32)
            p0[0][0] = [center[0],center[1]]

            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                if (abs(a - c) < 0.2 or abs(b - d) < 0.2):
                    continue
                cv2.line(cv_image, (a, b), (c, d), (0,0,255), 2)
                #cv2.circle(cv_image, (a, b), 5, (0,0,255)),
    
            #p0 = good_new.reshape(-1, 1, 2)
                                   
    cv2.imshow("yo", cv_image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        exit(0)

    old_gray = gray.copy() 
    

# update pointer to depth map
def depth_callback(img_msg):
    global depth_map
    try:
        depth_map = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except:
        rospy.logerr("CvBridge Error")

sub_image = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, image_callback)
sub_depth = rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, depth_callback)


while not rospy.is_shutdown():
    rospy.spin()