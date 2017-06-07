# CMakeLists file of Multiscale gigapixel video
#
# Author: Shane Yuan
# Date: May 23, 2017
#

cmake_minimum_required(VERSION 2.8)
project( MultiscaleGigaVideo )

if ( WIN32 )
	set(OpenCV_DIR "E:/ShaneYuan/lib/opencv/x64/vc12/lib")
	set(OpenCV_CONFIG_PATH "E:/ShaneYuan/lib/opencv/")
	set(WIN_HEADER_PATH "${PROJECT_SOURCE_DIR}/winheadfile")
    include_directories(${WIN_HEADER_PATH})
endif()
include_directories("${PROJECT_SOURCE_DIR}/common")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen3
if (WIN32)
	set(EIGEN_INCLUDE_DIR "E:/ShaneYuan/lib/eigen")
elseif ( UNIX )
		set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
endif()
include_directories(${EIGEN_INCLUDE_DIR})

file(GLOB HEADERS  *.h *.hpp "${PROJECT_SOURCE_DIR}/common/Global.h")
set(SRC_LIST "${PROJECT_SOURCE_DIR}/common/Global.cpp" "main.cpp" "GigaVideoIO.cpp"
"Quad.cpp" "Mesh.cpp" "ASAP.cpp" "ASAPadd.cpp" "ASAPwarp.cpp" "MultiscaleVideoStitcher.cpp"
"CrossResWarper.cpp" "RelightingInterpolation.cpp" "EdgeAwareInterpolator.cpp" "DeepFlow.cpp"
"Blender.cpp" ${HEADERS})

# OpenMP
if (WIN32)
	set(CMAKE_CXX_FLAGS_RELEASE "/openmp")
elseif( UNIX )
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

#c++11
if (UNIX)
    SET(CMAKE_CXX_FLAGS "-std=c++11")
endif()

add_executable( MultiscaleGigaVideo ${SRC_LIST} )

target_link_libraries( MultiscaleGigaVideo ${OpenCV_LIBS} )
