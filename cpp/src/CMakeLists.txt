cmake_minimum_required(VERSION 3.10)
project(yolo26depth-cpp VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV4 (RK3588 typically has opencv4 installed)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc)

# rknnrt library and headers (bundled in this directory since Rockchip
# does not always provide a dev package with CMake config)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    main.cpp
    rknn_model.cpp
    preprocess.cpp
    postprocess.cpp
)

# Build executable
add_executable(yolo26depth-cpp ${SOURCES})

target_link_libraries(yolo26depth-cpp
    PRIVATE
        rknnrt           # /usr/lib/librknnrt.so (board runtime)
        ${OpenCV_LIBS}
        stdc++fs         # C++17 filesystem (for older toolchains)
)

target_include_directories(yolo26depth-cpp
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

# Install target
install(TARGETS yolo26depth-cpp DESTINATION bin)
