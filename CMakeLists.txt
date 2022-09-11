cmake_minimum_required(VERSION 3.5)

project(image_localization_project)

set(OpenCV_DIR /Users/cameronfiore/C++/opencv/build)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(image_localization_project
        main.cpp
        src/sevenScenes.cpp
        src/synthetic.cpp
        src/Space.cpp
        src/functions.cpp
        src/poseEstimation.cpp
        src/OptimalRotationSolver.cpp
        src/CambridgeLandmarks.cpp
        src/aachen.cpp)

target_link_libraries(image_localization_project ${OpenCV_LIBS})
target_link_libraries(image_localization_project Ceres::ceres)
