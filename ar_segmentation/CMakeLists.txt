cmake_minimum_required(VERSION 3.10)
project(SuryaBench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add cmake directory to CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(CUDA REQUIRED)
find_package(CFITSIO REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)  # Find YAML-CPP

# Add include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CFITSIO_INCLUDE_DIRS})

# Add executable target
add_executable(SuryaBench main.cpp src/io.cpp src/utils.cpp)

# Link libraries
target_link_libraries(SuryaBench ${OpenCV_LIBS} ${HDF5_LIBRARIES} ${CUDA_LIBRARIES} ${CFITSIO_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} yaml-cpp)

# add config file location
add_definitions(-DCONFIG_PATH=\"${CMAKE_SOURCE_DIR}/config.yaml\")
