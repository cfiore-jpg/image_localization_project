cmake_minimum_required(VERSION 3.5)

project(ImageMatcherProject)

set(OpenCV_DIR /Users/cameronfiore/C++/build_opencv)
find_package(OpenCV REQUIRED)
set(DBoW2_DIR /Users/cameronfiore/C++/DBoW2/cmake-build-debug)
find_package(DBoW2 REQUIRED)
find_package(COLMAP REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Ceres REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/Users/cameronfiore/C++/DBoW2/include/DBoW2)
include_directories(${COLMAP_INCLUDE_DIRS})
link_directories(${COLMAP_LINK_DIRS})

add_executable(ImageMatcherProject main.cpp src/imageMatcher_Orb.cpp src/sevenScenes.cpp src/KDTree.cpp src/imageMatcher.cpp src/Calibrate.cpp include/processing.h)

target_link_libraries(ImageMatcherProject ${OpenCV_LIBS})
target_link_libraries(ImageMatcherProject /Users/cameronfiore/C++/DBoW2/cmake-build-debug/libDBoW2.dylib)
target_link_libraries(ImageMatcherProject ${COLMAP_LIBRARIES})
target_link_libraries(ImageMatcherProject ${OpenMP_CXX_LIBRARIES})
target_link_libraries(ImageMatcherProject Ceres::ceres)
