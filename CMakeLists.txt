cmake_minimum_required (VERSION 3.8)
project (StereoAlgorithmDemo)

# cuDNN path
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CUDNN_INCLUDE "/usr/include/aarch64-linux-gnu" CACHE INTERNAL "cuDNN include location")
    set(CUDNN_LIB "/usr/lib/aarch64-linux-gnu" CACHE INTERNAL "cuDNN lib location")
else()
    set(CUDNN_INCLUDE "/usr/include" CACHE INTERNAL "cuDNN include location")
    set(CUDNN_LIB "/usr/lib/x86_64-linux-gnu" CACHE INTERNAL "cuDNN lib location")
endif()

# tensorrt path
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(TensorRT_INCLUDE "/usr/include/aarch64-linux-gnu" CACHE INTERNAL "TensorRT Library include location")
    set(TensorRT_LIB "/usr/lib/aarch64-linux-gnu" CACHE INTERNAL "TensorRT Library lib location")
else()
    set(TensorRT_INCLUDE "/home/irl-19/newdisk/TensorRT-8.0.0.3/include" CACHE INTERNAL "TensorRT Library include location")
    set(TensorRT_LIB "/home/irl-19/newdisk/TensorRT-8.0.0.3/lib" CACHE INTERNAL "TensorRT Library lib location")
endif()

# 添加子目录
add_subdirectory("RAFTStereo")
add_subdirectory("Stereo_Calibration")
