 /*
  * OpenCV Example using ROS and CPP
  */

// Include the ROS library
#include <ros/ros.h>
#include <opencv_services/StringArray.h>

 // Include opencv2
//#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/video/background_segm.hpp"
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>


// Include CvBridge, Image Transport, Image msg
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/String.h>

// Include tf2 for transformation
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// zed stuff
#include <zed_interfaces/Object.h>
#include <zed_interfaces/ObjectsStamped.h>

#include "opencv_services/box_and_target_position.h"

using namespace std;
using namespace cv;
using namespace dnn;

// Topics
static const std::string IMAGE_TOPIC = "/zed/zed_node/right_raw/image_raw_color";
static const std::string POINT_CLOUD2_TOPIC = "/zed/zed_node/point_cloud/cloud_registered";

// Publisher
ros::Publisher pub;

tf2_ros::Buffer tf_buffer;

const std::string from_frame = "zed_right_camera_optical_frame";
const std::string to_frame = "base_link";

cv::Mat camera_image;

cv::Mat frame;
cv::Rect2d bbox(287, 23, 86, 320);
cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
bool first_run = true;

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;
 
// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
 
// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);

// trajectory tracking
cv::Mat imgLines;

void image_cb(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;

  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  float image_size_y = cv_ptr->image.rows;
  float image_size_x = cv_ptr->image.cols;

  namedWindow("Control", WINDOW_AUTOSIZE); //create a window called "Control"

  int iLowH = 0;
  int iHighH = 360;

  int iLowS = 0; 
  int iHighS = 100;

  int iLowV = 0;
  int iHighV = 100;

  //Create trackbars in "Control" window
  createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
  createTrackbar("HighH", "Control", &iHighH, 179);

  createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
  createTrackbar("HighS", "Control", &iHighS, 255);

  createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
  createTrackbar("HighV", "Control", &iHighV, 255);

  int iLastX = -1; 
  int iLastY = -1;

  //Capture a temporary image from the camera
  if (imgLines.empty())
  {
    //Create a black image with the size as the camera output
    imgLines = cv::Mat::zeros(cv_ptr->image.size(), CV_8UC3);;
  }
  


  cv::Mat imgOriginal;

  cv_ptr->image.copyTo(imgOriginal);

  if (imgOriginal.empty()) //if not success, break loop
  {
    return;
  }

  Mat imgHSV;

  cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

  Mat imgThresholded;

  inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
      
  //morphological opening (removes small objects from the foreground)
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
  dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

  //morphological closing (removes small holes from the foreground)
  dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

  //Calculate the moments of the thresholded image
  Moments oMoments = moments(imgThresholded);

  double dM01 = oMoments.m01;
  double dM10 = oMoments.m10;
  double dArea = oMoments.m00;

  if (dArea > 100)
  {
  //calculate the position of the ball
  int posX = dM10 / dArea;
  int posY = dM01 / dArea;        

  std::cout << posX << " " << posY << std::endl;
        
  if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
  {
    //Draw a red line from the previous point to the current point
    line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);
  }

  iLastX = posX;
  iLastY = posY;
  }

  imshow("Thresholded Image", imgThresholded); //show the thresholded image

  imgOriginal = imgOriginal + imgLines;
  imshow("Original", imgOriginal); //show the original image

  if (waitKey(30) == 27) 
  {
    exit(0);
  }
}

// tracking with ROI (trash)
void image_cb_roi_tracking(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;

  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  float image_size_y = cv_ptr->image.rows;
  float image_size_x = cv_ptr->image.cols;
  
  cv_ptr->image.copyTo(frame);

  if (first_run)
  {
    bbox = selectROI(frame, false); 
    tracker->init(frame, bbox);
    first_run = false;
  }

  rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 ); 

  // Start timer
  double timer = (double)getTickCount();
    
  // Update the tracking result
  bool ok = tracker->update(frame, bbox);
    
  // Calculate Frames per second (FPS)
  float fps = getTickFrequency() / ((double)getTickCount() - timer);
    
  if (ok)
  {
      // Tracking success : Draw the tracked object
      rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
  }
  else
  {
      // Tracking failure detected.
      putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
  }
    
  // Display tracker type on frame
  //putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
    
  // Display FPS on frame
  //putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

  // Display frame.
  imshow("frame", frame);
    
  // Exit if ESC pressed.
  int k = waitKey(1);
  if(k == 27)
  {
      exit(0);
  }
  
}

void objectListCallback(const zed_interfaces::ObjectsStamped::ConstPtr& msg)
{
  ROS_INFO("***** New object list *****");
  for (int i = 0; i < msg->objects.size(); i++)
  {
    if (msg->objects[i].label_id == -1)
      continue;

    ROS_INFO_STREAM(msg->objects[i].label << " [" << msg->objects[i].label_id << "] - Pos. ["
                                          << msg->objects[i].position[0] << "," << msg->objects[i].position[1] << ","
                                          << msg->objects[i].position[2] << "] [m]"
                                          << "- Conf. " << msg->objects[i].confidence
                                          << " - Tracking state: " << static_cast<int>(msg->objects[i].tracking_state));
  }
}

 // Main function
int main(int argc, char **argv)
{
  // The name of the node
  ros::init(argc, argv, "opencv_services");
   
  // Default handler for nodes in ROS
  ros::NodeHandle nh("");

    // Used to publish and subscribe to images
  image_transport::ImageTransport it(nh);

    // Subscribe to the /camera raw image topic
  image_transport::Subscriber image_sub = it.subscribe(IMAGE_TOPIC, 1, image_cb);
  tf2_ros::TransformListener listener(tf_buffer);
  //ros::Publisher pub = nh.advertise<std_msgs::String>("command", 1000);
  ros::Publisher pub = nh.advertise<opencv_services::StringArray>("command", 1000);

  bool send = true;
  int x = 1;

  while (ros::ok())
  {
  //std_msgs::String msg;
  opencv_services::StringArray msg;
  std::stringstream ss1;
  std::stringstream ss2;
  std::stringstream ss3;

  ss1 << "hello";
  ss2 << "goodbye";
  ss3 << "test";

  msg.strings.push_back(ss1.str());
  msg.strings.push_back(ss2.str());
  msg.strings.push_back(ss3.str());

    if (send)
    {
      pub.publish(msg);
      send = false;
    }

    if (x % 1000 == 0)
    {
      send = true;
      x = 0;
    }
    ros::spinOnce();
  }
  // object detection (requires zed2)
  //ros::Subscriber subObjList= nh.subscribe("/zed2/zed_node/obj_det/objects", 1, objectListCallback);
  
  // this causes graphical thrashing idk what else to call it, but it's bad
    // Subscribe to the /camera PointCloud2 topic
  //ros::Subscriber point_cloud_sub = nh.subscribe(POINT_CLOUD2_TOPIC, 1, point_cloud_cb);
  
  
   
  // ros::ServiceServer service = nh.advertiseService("box_and_target_position",  get_box_and_target_position);
   
  // Make sure we keep reading new video frames by calling the imageCallback function
  //ros::spin();
   
  // Close down OpenCV
  cv::destroyWindow("frame");
}

 

