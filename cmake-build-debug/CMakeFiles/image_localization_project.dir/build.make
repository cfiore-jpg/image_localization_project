# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/cameronfiore/C++/image_localization_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/cameronfiore/C++/image_localization_project/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/image_localization_project.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/image_localization_project.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/image_localization_project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/image_localization_project.dir/flags.make

CMakeFiles/image_localization_project.dir/main.cpp.o: CMakeFiles/image_localization_project.dir/flags.make
CMakeFiles/image_localization_project.dir/main.cpp.o: ../main.cpp
CMakeFiles/image_localization_project.dir/main.cpp.o: CMakeFiles/image_localization_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/image_localization_project.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/image_localization_project.dir/main.cpp.o -MF CMakeFiles/image_localization_project.dir/main.cpp.o.d -o CMakeFiles/image_localization_project.dir/main.cpp.o -c /Users/cameronfiore/C++/image_localization_project/main.cpp

CMakeFiles/image_localization_project.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_localization_project.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cameronfiore/C++/image_localization_project/main.cpp > CMakeFiles/image_localization_project.dir/main.cpp.i

CMakeFiles/image_localization_project.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_localization_project.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cameronfiore/C++/image_localization_project/main.cpp -o CMakeFiles/image_localization_project.dir/main.cpp.s

CMakeFiles/image_localization_project.dir/src/calibrate.cpp.o: CMakeFiles/image_localization_project.dir/flags.make
CMakeFiles/image_localization_project.dir/src/calibrate.cpp.o: ../src/calibrate.cpp
CMakeFiles/image_localization_project.dir/src/calibrate.cpp.o: CMakeFiles/image_localization_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/image_localization_project.dir/src/calibrate.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/image_localization_project.dir/src/calibrate.cpp.o -MF CMakeFiles/image_localization_project.dir/src/calibrate.cpp.o.d -o CMakeFiles/image_localization_project.dir/src/calibrate.cpp.o -c /Users/cameronfiore/C++/image_localization_project/src/calibrate.cpp

CMakeFiles/image_localization_project.dir/src/calibrate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_localization_project.dir/src/calibrate.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cameronfiore/C++/image_localization_project/src/calibrate.cpp > CMakeFiles/image_localization_project.dir/src/calibrate.cpp.i

CMakeFiles/image_localization_project.dir/src/calibrate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_localization_project.dir/src/calibrate.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cameronfiore/C++/image_localization_project/src/calibrate.cpp -o CMakeFiles/image_localization_project.dir/src/calibrate.cpp.s

CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.o: CMakeFiles/image_localization_project.dir/flags.make
CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.o: ../src/sevenScenes.cpp
CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.o: CMakeFiles/image_localization_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.o -MF CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.o.d -o CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.o -c /Users/cameronfiore/C++/image_localization_project/src/sevenScenes.cpp

CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cameronfiore/C++/image_localization_project/src/sevenScenes.cpp > CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.i

CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cameronfiore/C++/image_localization_project/src/sevenScenes.cpp -o CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.s

CMakeFiles/image_localization_project.dir/src/synthetic.cpp.o: CMakeFiles/image_localization_project.dir/flags.make
CMakeFiles/image_localization_project.dir/src/synthetic.cpp.o: ../src/synthetic.cpp
CMakeFiles/image_localization_project.dir/src/synthetic.cpp.o: CMakeFiles/image_localization_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/image_localization_project.dir/src/synthetic.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/image_localization_project.dir/src/synthetic.cpp.o -MF CMakeFiles/image_localization_project.dir/src/synthetic.cpp.o.d -o CMakeFiles/image_localization_project.dir/src/synthetic.cpp.o -c /Users/cameronfiore/C++/image_localization_project/src/synthetic.cpp

CMakeFiles/image_localization_project.dir/src/synthetic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_localization_project.dir/src/synthetic.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cameronfiore/C++/image_localization_project/src/synthetic.cpp > CMakeFiles/image_localization_project.dir/src/synthetic.cpp.i

CMakeFiles/image_localization_project.dir/src/synthetic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_localization_project.dir/src/synthetic.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cameronfiore/C++/image_localization_project/src/synthetic.cpp -o CMakeFiles/image_localization_project.dir/src/synthetic.cpp.s

CMakeFiles/image_localization_project.dir/src/Space.cpp.o: CMakeFiles/image_localization_project.dir/flags.make
CMakeFiles/image_localization_project.dir/src/Space.cpp.o: ../src/Space.cpp
CMakeFiles/image_localization_project.dir/src/Space.cpp.o: CMakeFiles/image_localization_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/image_localization_project.dir/src/Space.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/image_localization_project.dir/src/Space.cpp.o -MF CMakeFiles/image_localization_project.dir/src/Space.cpp.o.d -o CMakeFiles/image_localization_project.dir/src/Space.cpp.o -c /Users/cameronfiore/C++/image_localization_project/src/Space.cpp

CMakeFiles/image_localization_project.dir/src/Space.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_localization_project.dir/src/Space.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cameronfiore/C++/image_localization_project/src/Space.cpp > CMakeFiles/image_localization_project.dir/src/Space.cpp.i

CMakeFiles/image_localization_project.dir/src/Space.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_localization_project.dir/src/Space.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cameronfiore/C++/image_localization_project/src/Space.cpp -o CMakeFiles/image_localization_project.dir/src/Space.cpp.s

CMakeFiles/image_localization_project.dir/src/functions.cpp.o: CMakeFiles/image_localization_project.dir/flags.make
CMakeFiles/image_localization_project.dir/src/functions.cpp.o: ../src/functions.cpp
CMakeFiles/image_localization_project.dir/src/functions.cpp.o: CMakeFiles/image_localization_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/image_localization_project.dir/src/functions.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/image_localization_project.dir/src/functions.cpp.o -MF CMakeFiles/image_localization_project.dir/src/functions.cpp.o.d -o CMakeFiles/image_localization_project.dir/src/functions.cpp.o -c /Users/cameronfiore/C++/image_localization_project/src/functions.cpp

CMakeFiles/image_localization_project.dir/src/functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_localization_project.dir/src/functions.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cameronfiore/C++/image_localization_project/src/functions.cpp > CMakeFiles/image_localization_project.dir/src/functions.cpp.i

CMakeFiles/image_localization_project.dir/src/functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_localization_project.dir/src/functions.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cameronfiore/C++/image_localization_project/src/functions.cpp -o CMakeFiles/image_localization_project.dir/src/functions.cpp.s

CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.o: CMakeFiles/image_localization_project.dir/flags.make
CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.o: ../src/poseEstimation.cpp
CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.o: CMakeFiles/image_localization_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.o -MF CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.o.d -o CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.o -c /Users/cameronfiore/C++/image_localization_project/src/poseEstimation.cpp

CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cameronfiore/C++/image_localization_project/src/poseEstimation.cpp > CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.i

CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cameronfiore/C++/image_localization_project/src/poseEstimation.cpp -o CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.s

CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.o: CMakeFiles/image_localization_project.dir/flags.make
CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.o: ../src/OptimalRotationSolver.cpp
CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.o: CMakeFiles/image_localization_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.o -MF CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.o.d -o CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.o -c /Users/cameronfiore/C++/image_localization_project/src/OptimalRotationSolver.cpp

CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cameronfiore/C++/image_localization_project/src/OptimalRotationSolver.cpp > CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.i

CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cameronfiore/C++/image_localization_project/src/OptimalRotationSolver.cpp -o CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.s

CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.o: CMakeFiles/image_localization_project.dir/flags.make
CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.o: ../src/CambridgeLandmarks.cpp
CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.o: CMakeFiles/image_localization_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.o -MF CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.o.d -o CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.o -c /Users/cameronfiore/C++/image_localization_project/src/CambridgeLandmarks.cpp

CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cameronfiore/C++/image_localization_project/src/CambridgeLandmarks.cpp > CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.i

CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cameronfiore/C++/image_localization_project/src/CambridgeLandmarks.cpp -o CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.s

# Object files for target image_localization_project
image_localization_project_OBJECTS = \
"CMakeFiles/image_localization_project.dir/main.cpp.o" \
"CMakeFiles/image_localization_project.dir/src/calibrate.cpp.o" \
"CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.o" \
"CMakeFiles/image_localization_project.dir/src/synthetic.cpp.o" \
"CMakeFiles/image_localization_project.dir/src/Space.cpp.o" \
"CMakeFiles/image_localization_project.dir/src/functions.cpp.o" \
"CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.o" \
"CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.o" \
"CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.o"

# External object files for target image_localization_project
image_localization_project_EXTERNAL_OBJECTS =

image_localization_project: CMakeFiles/image_localization_project.dir/main.cpp.o
image_localization_project: CMakeFiles/image_localization_project.dir/src/calibrate.cpp.o
image_localization_project: CMakeFiles/image_localization_project.dir/src/sevenScenes.cpp.o
image_localization_project: CMakeFiles/image_localization_project.dir/src/synthetic.cpp.o
image_localization_project: CMakeFiles/image_localization_project.dir/src/Space.cpp.o
image_localization_project: CMakeFiles/image_localization_project.dir/src/functions.cpp.o
image_localization_project: CMakeFiles/image_localization_project.dir/src/poseEstimation.cpp.o
image_localization_project: CMakeFiles/image_localization_project.dir/src/OptimalRotationSolver.cpp.o
image_localization_project: CMakeFiles/image_localization_project.dir/src/CambridgeLandmarks.cpp.o
image_localization_project: CMakeFiles/image_localization_project.dir/build.make
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_gapi.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_stitching.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_alphamat.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_aruco.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_bgsegm.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_bioinspired.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_ccalib.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_dnn_objdetect.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_dnn_superres.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_dpm.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_face.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_fuzzy.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_hfs.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_img_hash.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_intensity_transform.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_line_descriptor.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_mcc.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_quality.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_rapid.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_reg.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_rgbd.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_saliency.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_sfm.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_stereo.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_structured_light.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_superres.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_surface_matching.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_tracking.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_videostab.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_xfeatures2d.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_xobjdetect.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_xphoto.4.5.0.dylib
image_localization_project: /opt/homebrew/lib/libceres.2.1.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_highgui.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_shape.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_datasets.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_plot.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_text.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_dnn.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_ml.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_phase_unwrapping.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_optflow.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_ximgproc.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_video.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_videoio.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_imgcodecs.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_objdetect.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_calib3d.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_features2d.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_flann.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_photo.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_imgproc.4.5.0.dylib
image_localization_project: /Users/cameronfiore/C++/opencv/build/lib/libopencv_core.4.5.0.dylib
image_localization_project: /opt/homebrew/lib/libglog.0.6.0.dylib
image_localization_project: /opt/homebrew/lib/libgflags.2.2.2.dylib
image_localization_project: CMakeFiles/image_localization_project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable image_localization_project"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/image_localization_project.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/image_localization_project.dir/build: image_localization_project
.PHONY : CMakeFiles/image_localization_project.dir/build

CMakeFiles/image_localization_project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/image_localization_project.dir/cmake_clean.cmake
.PHONY : CMakeFiles/image_localization_project.dir/clean

CMakeFiles/image_localization_project.dir/depend:
	cd /Users/cameronfiore/C++/image_localization_project/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/cameronfiore/C++/image_localization_project /Users/cameronfiore/C++/image_localization_project /Users/cameronfiore/C++/image_localization_project/cmake-build-debug /Users/cameronfiore/C++/image_localization_project/cmake-build-debug /Users/cameronfiore/C++/image_localization_project/cmake-build-debug/CMakeFiles/image_localization_project.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/image_localization_project.dir/depend

