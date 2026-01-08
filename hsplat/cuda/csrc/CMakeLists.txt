cmake_minimum_required(VERSION 3.12)
project(cuda_extension LANGUAGES CXX CUDA)

# Ensure Position Independent Code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and Torch
find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Include directories
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

# Collect source files
file(GLOB_RECURSE CUDA_SRCS ${CMAKE_SOURCE_DIR}/*.cu ${CMAKE_SOURCE_DIR}/*.cpp)

# Create shared library
add_library(cuda_extension MODULE ${CUDA_SRCS})

# Set the output properties
set_target_properties(cuda_extension PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Link libraries
target_link_libraries(cuda_extension ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
