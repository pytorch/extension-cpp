cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX CUDA)

# Set the C++ standard for all targets
set(CMAKE_CXX_STANDARD 20) # This might be unsafe since pytorch use std17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable better clangd support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
        COMMAND "${Python3_EXECUTABLE}" "-c" "import torch;print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE PT_CMAKE_PREFIX
        COMMAND_ECHO STDOUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
)

# cache CUDA_ARCHITECTURES, which seems to be reset by Torch
set(TMP_STORE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${PT_CMAKE_PREFIX})

find_package(Torch REQUIRED CONFIG)

#  simple_cuda source files
file(GLOB_RECURSE CU_SOURCES csrc/*.cu)
file(GLOB_RECURSE CPP_SOURCES csrc/*.cpp)
MESSAGE(STATUS "CU_SOURCES: ${CU_SOURCES}")
MESSAGE(STATUS "CPP_SOURCES: ${CPP_SOURCES}")

add_library(${SKBUILD_PROJECT_NAME} SHARED
  ${CU_SOURCES}
  ${CPP_SOURCES}
)

# Set the library output directory, I think this makes ninja builds work
set_target_properties(${SKBUILD_PROJECT_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${SKBUILD_PROJECT_NAME}/lib"
)
# Add include directories to the library
target_include_directories(${SKBUILD_PROJECT_NAME} PUBLIC src/include)

# Link the library to the Torch library
target_link_libraries(${SKBUILD_PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} Python::Python)

# Install the library to the wheel distribution
install(TARGETS ${SKBUILD_PROJECT_NAME}
  LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/lib
)
