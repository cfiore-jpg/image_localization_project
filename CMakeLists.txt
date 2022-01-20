cmake_minimum_required(VERSION 3.5)

project(image_localization_project)

set(OpenCV_DIR /Users/cameronfiore/C++/opencv/build)
find_package(OpenCV REQUIRED)
set(DBoW2_DIR /Users/cameronfiore/C++/DBoW2/build)
find_package(DBoW2 REQUIRED)
set(DBoW3_DIR /Users/cameronfiore/C++/DBoW3/build)
find_package(DBoW3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Ceres REQUIRED)
set(CMAKE_CXX_STANDARD 14)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/Users/cameronfiore/C++/DBoW2/include/DBoW2)
include_directories(/Users/cameronfiore/C++/DBoW3/src)

add_executable(image_localization_project main.cpp src/calibrate.cpp src/imageMatcher_Orb.cpp src/sevenScenes.cpp src/Space.cpp src/imageMatcher.cpp src/functions.cpp include/calibrate.h)

target_link_libraries(image_localization_project ${OpenCV_LIBS})
target_link_libraries(image_localization_project /Users/cameronfiore/C++/DBoW2/build/libDBoW2.dylib)
target_link_libraries(image_localization_project /Users/cameronfiore/C++/DBow3/build/src/libDBoW3.dylib)
target_link_libraries(image_localization_project Ceres::ceres)
