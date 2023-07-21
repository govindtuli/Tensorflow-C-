cmake_minimum_required(VERSION 3.10)
project(tensorflow_cpp_project)

set(CMAKE_CXX_STANDARD 14)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Set the path to the TensorFlow package
# Be sure to replace this with your actual path
set(TensorFlow_DIR /tensorflow-cpp)  

# Find TensorFlow package
find_package(TensorFlow REQUIRED)

# Add TensorFlow include directories
include_directories(${TensorFlow_INCLUDE_DIRS})

# Add your source files
add_executable(tensorflow_cpp_app main.cpp)

# Link against TensorFlow and OpenCV libraries
target_link_libraries(tensorflow_cpp_app ${TensorFlow_LIBRARIES} ${OpenCV_LIBS})




