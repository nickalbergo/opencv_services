#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import cv2
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('ros_python_cv', anonymous=True)
pub = rospy.Publisher('ee_center', Point, queue_size=10)

bridge = CvBridge()

face_cascade = cv2.CascadeClassifier("haar_cascade.xml")

tracker = cv2.TrackerCSRT_create()
bbox = (287, 23, 86, 320)
init_tracker = True

def image_callback(img_msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except:
        rospy.logerr("CvBridge Error")

    # convert to gray scale of each frames
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    global init_tracker, bbox

    if (init_tracker):
        bbox = cv2.selectROI(cv_image, False)
        tracker.init(gray, bbox)
        init_tracker = False

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
            print(center)
            c = Point(center[0], center[1], 0)
            pub.publish(c)

    cv2.imshow("yo", cv_image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        exit(0)

sub_image = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, image_callback)

while not rospy.is_shutdown():
    rospy.spin()