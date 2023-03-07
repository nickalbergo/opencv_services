 /*
  * OpenCV Example using ROS and CPP
  */

 // Include the ROS library
 #include <ros/ros.h>

 // Include opencv2
 //#include <opencv2/core/mat.hpp>
  #include <opencv2/core.hpp>
  #include <opencv2/highgui.hpp>
  #include <opencv2/imgproc.hpp>
  #include "opencv2/video/background_segm.hpp"
  //#include <opencv2/highgui.hpp>
  //#include <opencv2/imgproc.hpp>
  #include <opencv2/videoio.hpp>
  #include <opencv2/video.hpp>
 // Include CvBridge, Image Transport, Image msg
 #include <image_transport/image_transport.h>
 #include <cv_bridge/cv_bridge.h>
 #include <sensor_msgs/image_encodings.h>
 #include <sensor_msgs/PointCloud2.h>
 #include <geometry_msgs/Point.h>
 #include <geometry_msgs/PoseStamped.h>
 #include <geometry_msgs/Pose.h>

// Include tf2 for transformation
 #include <tf2_ros/buffer.h>
 #include <tf2_ros/transform_listener.h>
 #include <tf2_geometry_msgs/tf2_geometry_msgs.h>

 #include "opencv_services/box_and_target_position.h"

using namespace std;
using namespace cv;
// Topics
static const std::string IMAGE_TOPIC = "/zed/zed_node/right_raw/image_raw_color";
static const std::string POINT_CLOUD2_TOPIC = "/zed/zed_node/point_cloud/cloud_registered";

// Publisher
ros::Publisher pub;

tf2_ros::Buffer tf_buffer;

const std::string from_frame = "zed_right_camera_optical_frame";
const std::string to_frame = "base_link";

cv::Mat camera_image;

cv::Point2f box_centroid;
cv::Point2f target_centroid;

geometry_msgs::Point box_position_base_frame;
geometry_msgs::Point target_position_base_frame;

cv::Point2f search_centroid_in_area(std::vector<cv::Point2f> centroid_vector, cv::Rect area) {
  float sum_x = 0.0;
  float sum_y = 0.0;
  int number_of_centroids_in_area = 0;
  
  for( int i = 0; i<centroid_vector.size(); i++) {
    if(centroid_vector[i].inside(area)) {
      sum_x += centroid_vector[i].x;
      sum_y += centroid_vector[i].y;
      number_of_centroids_in_area++;
    }
  }
  cv::Point2f extracted_point(sum_x/number_of_centroids_in_area, sum_y/number_of_centroids_in_area);
  return extracted_point;
}

cv::Mat apply_cv_algorithms(cv::Mat camera_image) {
  // convert the image to grayscale format
  cv::Mat img_gray;
  cv::cvtColor(camera_image, img_gray, cv::COLOR_BGR2GRAY);

  // GaussianBlur(img_gray, img_gray, Size(21, 21), 0)

  cv::Mat canny_output;
  cv::Canny(img_gray,canny_output,10,350);

  return canny_output;

  //return img_gray;
}

std::vector<cv::Point2f> extract_centroids(cv::Mat canny_output) {

  // detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
   std::vector<std::vector<cv::Point>> contours;
   std::vector<cv::Vec4i> hierarchy;
   cv::findContours(canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

  // get the moments
  std::vector<cv::Moments> mu(contours.size());
  for( int i = 0; i<contours.size(); i++ )
  { mu[i] = cv::moments( contours[i], false ); }
  
  // get the centroid of figures.
  std::vector<cv::Point2f> centroids(contours.size());
  for( int i = 0; i<contours.size(); i++) {
    float centroid_x = mu[i].m10/mu[i].m00;
    float centroid_y = mu[i].m01/mu[i].m00;
    centroids[i] = cv::Point2f(centroid_x, centroid_y);
  }

    // draw contours
  cv::Mat drawing(canny_output.size(), CV_8UC3, cv::Scalar(255,255,255));

  for( int i = 0; i<contours.size(); i++ )
  {
  cv::Scalar color = cv::Scalar(167,151,0); // B G R values
  cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
  cv::circle( drawing, centroids[i], 4, color, -1, 8, 0 );
  }

   // show the resuling image
  cv::namedWindow( "Extracted centroids", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Extracted centroids", drawing );
  cv::waitKey(3);

  return centroids;
}

// cv::Scalar color_mint(255, 255, 0);
// cv::Mat previous_frame;
// cv::Mat frame1, prvs;
// cv::Mat frame2, next;
// cv::Mat old_frame, old_gray;
// vector<Point2f> p0, p1;
// set first frame
cv::Mat frame1, prvs;
cv::Mat old_frame, old_gray, mask;
std::vector<cv::Point2f> p0, p1;
vector<cv::Scalar> colors;
bool USE_SPARSE = true;

void image_cb(const sensor_msgs::ImageConstPtr& msg)
{
  // cv::Mat frame_delta;
  // cv::Mat thresh;
  // cv::Mat frame;
  // cv::Mat output;
  // std::vector<std::vector<cv::Point> > cnts;
  cv_bridge::CvImagePtr cv_ptr;
  // cv::Mat mask;

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

  if (!USE_SPARSE)
  {
    if (frame1.empty())
    {
      cv_ptr->image.copyTo(frame1);
      cv::cvtColor(frame1, prvs, COLOR_BGR2GRAY);
    }

    Mat frame2, next;
    cv_ptr->image.copyTo(frame2);

    if (frame2.empty())
        return;

    cvtColor(frame2, next, COLOR_BGR2GRAY);

    cv::Mat flow(prvs.size(), CV_32FC2);
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    // visualization
    cv::Mat flow_parts[2];
    split(flow, flow_parts);
    cv::Mat magnitude, angle, magn_norm;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    //build hsv image
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);

    imshow("frame", bgr);

    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27)
        exit(0);

    prvs = next;

  }
  else
  {
    if (old_frame.empty())
    {
      cv_ptr->image.copyTo(old_frame);
      cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
      goodFeaturesToTrack(old_gray, p0, 100, 0.3, 1, Mat()); //, 7, false, 0.04);
      cv::Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
      //cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    }

    cv::Mat frame, frame_gray;

    cv_ptr->image.copyTo(frame);

    if (frame.empty())
        return;

    cv::cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    // calculate optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    //cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
    cv::calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err); //, Size(15,15), 2, criteria);

    std::vector<Point2f> good_new;

    for(uint i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            good_new.push_back(p1[i]);
            // draw the tracks
            line(frame,p1[i], p0[i], colors[0], 2);
            //circle(frame, p1[i], 5, colors[i], -1);
            //std::cout << "from " << p0[i] << " to " << p1[i] << std::endl; 
            // if (p1[i].x > p0[i].x)
            // {
            //   std::cout << "x1 > x0" << std::endl;
            // }
            // if (p1[i].x < p0[i].x)
            // {
            //   std::cout << "x1 < x0" << std::endl;
            // }
            // if (p1[i].y > p0[i].y)
            // {
            //   std::cout << "y1 > y0" << std::endl;
            // }
            // if (p1[i].y < p0[i].y)
            // {
            //   std::cout << "y1 < y0" << std::endl;
            // }
        }
    }

    //cv::Mat img;
    //std::cout << frame.channels() << " " << mask.channels() << " " << img.channels() << std::endl;

    // if (mask.empty() || cv::countNonZero(mask) < 1)
    //   return;

    // if (!mask.empty())
    //   cv::add(frame, mask, frame);

    cv::imshow("frame", frame);

    int keyboard = cv::waitKey(30);
    if (keyboard == 'q' || keyboard == 27)
        exit(0);

    // Now update the previous frame and previous points
    frame_gray.copyTo(old_gray);
    p0 = good_new;
  }



  // cv_ptr->image.copyTo(output);
  // cv::cvtColor(cv_ptr->image, frame, cv::COLOR_BGR2GRAY);

  // if (frame.empty())
  //   return;
  
  // frame.convertTo(frame, CV_8U, 1 / 256.0);

  // vector<Scalar> colors;
  // RNG rng;
  // for(int i = 0; i < 100; i++)
  // {
  //     int r = rng.uniform(0, 256);
  //     int g = rng.uniform(0, 256);
  //     int b = rng.uniform(0, 256);
  //     colors.push_back(Scalar(r,g,b));
  // }
  
  
  // // Take first frame and find corners in it
  // if (old_frame.empty())
  // {
  //   frame.copyTo(old_frame);
  //   old_frame.copyTo(old_gray);
  //   //cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
  //   goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
  //   // Create a mask image for drawing purposes
  //   mask = Mat::zeros(old_frame.size(), old_frame.type());
  // }

  // //while(true){
  //     Mat current, frame_gray;
  //     frame.copyTo(current);
  //     if (current.empty())
  //         return;
  //     frame.copyTo(frame_gray);

  //     // cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
  //     // calculate optical flow
  //     vector<uchar> status;
  //     vector<float> err;
  //     TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
  //     calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
  //     vector<Point2f> good_new;
  //     for(uint i = 0; i < p0.size(); i++)
  //     {
  //         // Select good points
  //         if(status[i] == 1) {
  //             good_new.push_back(p1[i]);
  //             // draw the tracks
  //             line(mask,p1[i], p0[i], colors[i], 2);
  //             circle(current, p1[i], 5, colors[i], -1);
  //         }
  //     }
  //     Mat img;
  //     add(current, mask, img);
  //     imshow("Frame", img);
  //     int keyboard = waitKey(30);
  //     if (keyboard == 'q' || keyboard == 27)
  //         return;
  //     // Now update the previous frame and previous points
  //     //old_gray = frame_gray.clone();
  //     frame_gray.copyTo(old_gray);
  //     p0 = good_new;
  // //}











  // frame.copyTo(frame1);
  // //cv::cvtColor(frame1, prvs, cv::COLOR_BGR2GRAY);

  // //while(true){
  
  // frame.copyTo(frame2);
  // if (frame2.empty())
  //     return;
  // //cv::cvtColor(frame2, next, cv::COLOR_BGR2GRAY);
  // cv::Mat flow(prvs.size(), CV_32FC2);
  // cv::calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
  // // visualization
  // cv::Mat flow_parts[2];
  // cv::split(flow, flow_parts);
  // cv::Mat magnitude, angle, magn_norm;
  // cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
  // cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
  // angle *= ((1.f / 360.f) * (180.f / 255.f));
  // //build hsv image
  // cv::Mat _hsv[3], hsv, hsv8, bgr;
  // _hsv[0] = angle;
  // _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
  // _hsv[2] = magn_norm;
  // cv::merge(_hsv, 3, hsv);
  // hsv.convertTo(hsv8, CV_8U, 255.0);
  // cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
  // cv::imshow("frame2", bgr);
  // int keyboard = cv::waitKey(30);
  // if (keyboard == 'q' || keyboard == 27)
  //     exit(0);
  // prvs.copyTo(next);
  // //}



  // cv::Canny(frame,frame,10,350);

  // if (previous_frame.empty())
  //   frame.copyTo(previous_frame);
  
  // GaussianBlur(frame, frame, cv::Size(21, 21), 0);

  // //compute difference between first frame and current frame
  // absdiff(previous_frame, frame, frame_delta);
  // threshold(frame_delta, thresh, 60, 255, cv::THRESH_BINARY);
  
  // dilate(thresh, thresh, cv::Mat(), cv::Point(-1,-1), 2);
  // findContours(thresh, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // for(int i = 0; i< cnts.size(); i++) {
  //     if(contourArea(cnts[i]) < 500) {
  //         continue;
  //     }


  //   drawContours(output, std::vector<std::vector<cv::Point> >(1,cnts[i]), -1, color_mint, 1, 8);
  //     //putText(frame, "Motion Detected", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,255),2);
  // }


  // // Display the resulting frame
  // imshow( "output", output );
  
  // //imshow( "output", output );
  // // imshow( "Frame2", cv_ptr->image );
  
  // // frame.copyTo(previous_frame);

  // // Press  ESC on keyboard to exit
  // char c=(char)cv::waitKey(25);

  // if(c==27)
  //   exit(0); 


  //std::vector<cv::Point2f> centroids = extract_centroids(canny_output);

  //get box location in 2d image
  //cv::Rect box_search_area((image_size_x/2), 0, (image_size_x/2), 255);
  //box_centroid = search_centroid_in_area(centroids, box_search_area);

  //get plate location in 2d image
  //cv::Rect target_search_area(0, 0, (image_size_x/2), 255);
  //target_centroid = search_centroid_in_area(centroids, target_search_area);

}


geometry_msgs::Point pixel_to_3d_point(const sensor_msgs::PointCloud2 pCloud, const int u, const int v)
{
  // get width and height of 2D point cloud data
  int width = pCloud.width;
  int height = pCloud.height;

  // Convert from u (column / width), v (row/height) to position in array
  // where X,Y,Z data starts
  int arrayPosition = v*pCloud.row_step + u*pCloud.point_step;

  // compute position in array where x,y,z data start
  int arrayPosX = arrayPosition + pCloud.fields[0].offset; // X has an offset of 0
  int arrayPosY = arrayPosition + pCloud.fields[1].offset; // Y has an offset of 4
  int arrayPosZ = arrayPosition + pCloud.fields[2].offset; // Z has an offset of 8

  float X = 0.0;
  float Y = 0.0;
  float Z = 0.0;

  memcpy(&X, &pCloud.data[arrayPosX], sizeof(float));
  memcpy(&Y, &pCloud.data[arrayPosY], sizeof(float));
  memcpy(&Z, &pCloud.data[arrayPosZ], sizeof(float));

  geometry_msgs::Point p;
  p.x = X;
  p.y = Y;
  p.z = Z;

  return p;
}

geometry_msgs::Point transform_between_frames(geometry_msgs::Point p, const std::string from_frame, const std::string to_frame) {
    
  geometry_msgs::PoseStamped input_pose_stamped;
  input_pose_stamped.pose.position = p;
  input_pose_stamped.header.frame_id = from_frame;
  input_pose_stamped.header.stamp = ros::Time::now();

  geometry_msgs::PoseStamped output_pose_stamped = tf_buffer.transform(input_pose_stamped, to_frame, ros::Duration(1));
  return output_pose_stamped.pose.position;
}

void point_cloud_cb(const sensor_msgs::PointCloud2 pCloud) {
  // geometry_msgs::Point box_position_camera_frame;
  // box_position_camera_frame = pixel_to_3d_point(pCloud, box_centroid.x, box_centroid.y);

  // geometry_msgs::Point target_position_camera_frame;
  // target_position_camera_frame = pixel_to_3d_point(pCloud, target_centroid.x, target_centroid.y);
    
  // box_position_base_frame = transform_between_frames(box_position_camera_frame, from_frame, to_frame);
  // target_position_base_frame = transform_between_frames(target_position_camera_frame, from_frame, to_frame);

  // ROS_INFO_STREAM("3d box position base frame: x " << box_position_base_frame.x << " y " << box_position_base_frame.y << " z " << box_position_base_frame.z);
  // ROS_INFO_STREAM("3d target position base frame: x " << target_position_base_frame.x << " y " << target_position_base_frame.y << " z " << target_position_base_frame.z);
}

// service call response
bool get_box_and_target_position(opencv_services::box_and_target_position::Request  &req,
    opencv_services::box_and_target_position::Response &res) {
      res.box_position = box_position_base_frame;
      res.target_position = target_position_base_frame;
      return true;
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

  // global
  colors.push_back(cv::Scalar(255, 255, 0));

    // Subscribe to the /camera raw image topic
  image_transport::Subscriber image_sub = it.subscribe(IMAGE_TOPIC, 1, image_cb);

  // this causes graphical thrashing idk what else to call it, but it's bad
    // Subscribe to the /camera PointCloud2 topic
  //ros::Subscriber point_cloud_sub = nh.subscribe(POINT_CLOUD2_TOPIC, 1, point_cloud_cb);
  
  tf2_ros::TransformListener listener(tf_buffer);
   
  ros::ServiceServer service = nh.advertiseService("box_and_target_position",  get_box_and_target_position);
   
  // Make sure we keep reading new video frames by calling the imageCallback function
  ros::spin();
   
  // Close down OpenCV
  cv::destroyWindow("frame");
}

 

