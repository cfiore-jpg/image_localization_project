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
find_package(BLAS)
find_package(LAPACK)

set(CMAKE_CXX_STANDARD 14)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/opt/homebrew/Cellar/python@3.9/3.9.9/Frameworks/Python.framework/Versions/3.9/include/python3.9)
include_directories(/opt/homebrew/lib/python3.9/site-packages/numpy/core/include/numpy)
include_directories(/Users/cameronfiore/C++/matplotlib-cpp)
include_directories(/Users/cameronfiore/C++/DBoW2/include/DBoW2)
include_directories(/Users/cameronfiore/C++/DBoW3/src)
INCLUDE_DIRECTORIES(
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/5point
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/matrix
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/imagelib
        /Users/cameronfiore/C++/bundler-v0.4-source/include
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/cblas
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/minpack)

SET(CODEFILE
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/matrix/matrix.c
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/matrix/vector.c
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/matrix/svd.c
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/imagelib/triangulate.c
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/imagelib/fmatrix.c)

add_executable(image_localization_project
        main.cpp
        src/calibrate.cpp
        src/imageMatcher_Orb.cpp
        src/sevenScenes.cpp
        src/synthetic.cpp
        src/Space.cpp
        src/imageMatcher.cpp
        src/functions.cpp
        src/poseEstimation.cpp
        src/OptimalRotationSolver.cpp
        src/CambridgeLandmarks.cpp
        ${CODEFILE})

target_link_libraries(image_localization_project ${OpenCV_LIBS})
target_link_libraries(image_localization_project /Users/cameronfiore/C++/DBoW2/build/libDBoW2.dylib)
target_link_libraries(image_localization_project /Users/cameronfiore/C++/DBow3/build/src/libDBoW3.dylib)
target_link_libraries(image_localization_project Ceres::ceres)
target_link_libraries(image_localization_project /opt/homebrew/Cellar/python@3.9/3.9.9/Frameworks/Python.framework/Versions/3.9/lib/libpython3.9.dylib)
TARGET_LINK_LIBRARIES(image_localization_project
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/libimage.a
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/libminpack.a
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/libcblas.a
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/libf2c.a
        /Users/cameronfiore/C++/bundler-v0.4-source/lib/lib5point.a
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES})
