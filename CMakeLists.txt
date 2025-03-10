cmake_minimum_required(VERSION 3.0.2)
project(opencv_services)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  tf2_ros
  message_generation
  zed_interfaces
)

find_package(OpenCV 4 REQUIRED)

## Generate services in the 'srv' folder
 add_service_files(
   FILES
   box_and_target_position.srv
   predicted_joints.srv
 )

add_message_files(
  FILES
  StringArray.msg
  PointList.msg
)

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
   geometry_msgs
   sensor_msgs
 )



catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS roscpp
                 sensor_msgs
                 std_msgs
                 cv_bridge
                 image_transport
                 tf2_ros
                 message_runtime
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(opencv_extract_object_positions src/opencv_extract_object_positions.cpp)
add_dependencies(opencv_extract_object_positions opencv_services_generate_messages_cpp)

add_executable(opencv_testing src/opencv_testing.cpp)
add_dependencies(opencv_testing opencv_services_generate_messages_cpp)

add_executable(service_testing src/service_testing.cpp)
add_dependencies(service_testing opencv_services_generate_messages_cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(opencv_extract_object_positions ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(opencv_testing ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(service_testing ${catkin_LIBRARIES} ${OpenCV_LIBS})
