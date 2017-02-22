cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

LIST(APPEND CMAKE_MODULE_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(caffe_intel)

# ---[ Warfel build environment customizations
# find compiler
#SET(Env{CMAKE_CC_COMPILE} "mpicc")
#SET(Env{CMAKE_CXX_COMPILE} "mpicxx")
#SET(Env{CC} "mpicc")
#SET(Env{CXX} "mpicxx")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI=1")

message(STATUS "CMAKE CXX Compiler ID is ${CMAKE_CXX_COMPILER_ID} ")
message(STATUS "CMAKE C Compiler ID is ${CMAKE_C_COMPILER_ID} ")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message("-- Compiler is clang(++)")
    set(CMAKE_COMPILER_IS_CLANG 1)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message("-- Compiler is GNU ")
    set(CMAKE_COMPILER_IS_GNUCXX 1)
    set(CMAKE_COMPILER_IS_GNUCCC 1)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message("-- Compiler is ICC ")
    set(CMAKE_COMPILER_IS_ICC 1)
endif()


### Verify C and C++ compilers support c11 / c++11
#Verify C++ compiler supports C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


#Verify C compiler supports C11
include(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG("-std=c11" COMPILER_SUPPORTS_C11)
if(COMPILER_SUPPORTS_C11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 ")
else()
    message(STATUS "The compiler ${CMAKE_C_COMPILER_ID} has no C11 support. Please use a different C compiler.")
endif()
#change default build type to Release (for speed) rather than Debug
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Define build type, options are: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif(NOT CMAKE_BUILD_TYPE)


# set default compiler flags
set(USE_FLAGS " -Wall -Wno-format -fomit-frame-pointer -fstrict-aliasing ")
#for profiling
#set(USE_FLAGS " -Wall -Wno-format -fstrict-aliasing -pg ")

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(USE_FLAGS "${USE_FLAGS} -O1 -g")
endif(CMAKE_BUILD_TYPE MATCHES "Debug")


if (CMAKE_BUILD_TYPE MATCHES "Release")
    set(USE_FLAGS "${USE_FLAGS} -O3")
endif(CMAKE_BUILD_TYPE MATCHES "Release")


if (CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    set(USE_FLAGS "${USE_FLAGS} -O3")
endif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")



#minor optimizations on x64 for GCC variants
if (CMAKE_COMPILER_IS_GNUCXX)
    set(USE_FLAGS "${USE_FLAGS} -malign-double -fprefetch-loop-arrays")
endif(CMAKE_COMPILER_IS_GNUCXX)


# check if SSE ( and AVX?) features are present
INCLUDE(CheckSSEFeatures)
message(STATUS "SSE_FLAGS is ${SSE_FLAGS}")
set(USE_FLAGS "${USE_FLAGS} ${SSE_FLAGS}")


#set( ENV{LEVELDB_ROOT} "/home/files0/warf949/caffe3/leveldb/install")
set( ENV{LEVELDB_ROOT} "/home/d3n000/leveldb-1.18")

set( ENV{PYTHON_INCLUDE} "/home/d3n000/local_caffe/include/python2.7 $ENV{PYTHON_INCLUDE}")
set( ENV{PYTHON_INCLUDE}
        "/home/d3n000/local_caffe/lib/python2.7/site-packages/numpy-1.11.0-py2.7-linux-x86_64.egg/numpy/core/include/numpy $ENV{PYTHON_INCLUDE}")

set( PYTHON_LIBRARY "/home/d3n000/local_caffe/lib/libpython2.7.so")

message(STATUS "+++++ Initial CMAKE_PREFIX_PATH is ${CMAKE_PREFIX_PATH}")
set( CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/files0/warf949/caffe3/caffe-intel/cmake/Modules")
set( CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/files0/warf949/caffe3/leveldb/install")

set( LevelDB_DIR "/home/files0/warf949/caffe3/leveldb/install")
set( LevelDB_ROOT_DIR "/home/files0/warf949/caffe3/leveldb/install")

set( snappy_DIR "/home/files0/warf949/caffe3/snappy/install")
set( SNAPPY_ROOT_DIR "/home/files0/warf949/caffe3/snappy/install")

set(ENV{LMDB_DIR} "/home/d3n000/local_caffe")
set( BOOST_ROOT "/home/d3n000/boost_1_61_0")
set(GFLAGS_LIBRARY "/home/d3n000/local_caffe/lib/libgflags.so")
set(GFLAGS_INCLUDE_DIR "/home/d3n000/local_caffe/include")

set(GLOG_INCLUDE_DIR "/home/d3n000/local_caffe/include")
set(GLOG_LIBRARY "/home/d3n000/local_caffe/lib/libglog.so")

#set(PROTOBUF_ROOT "/home/d3n000/local_caffe")
set(PROTOBUF_ROOT_DIR "/home/d3n000/local_caffe")
set(PROTOBUF_INCLUDE_DIR "/home/d3n000/local_caffe/include")
set(PROTOBUF_LIBRARY "/home/d3n000/local_caffe/lib/libprotobuf.so")

#set(HDF5_hdf5_cpp_LIBRARY "/home/d3n000/local_caffe/lib/libhdf5.so")
#set(HDF5_INCLUDE_DIRS "/home/d3n000/local_caffe/include")
set(HDF5_ROOT "/home/d3n000/local_caffe")
#set(HDF5_LIBRARY "/home/d3n000/local_caffe/lib/libhdf5.so")
#set(HDF5_HL_LIBRARY "/home/d3n000/local_caffe/lib/libhdf5_hl.so")



# ---[ Caffe project
project(Caffe C CXX)

# ---[ Caffe version
set(CAFFE_TARGET_VERSION "1.0.0-rc3" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0-rc3" CACHE STRING "Caffe soname version")
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Options
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_OPENMP "Build Caffe with OpenMP support" ON )
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(USE_MKL2017_AS_DEFAULT_ENGINE "Use MKL2017 primitives for supported layers" OFF)
caffe_option(USE_MKLDNN_AS_DEFAULT_ENGINE "Use MKL-DNN primitives for supported layers" OFF)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
caffe_option(USE_SYSTEMTAP "Build for SystemTap" OFF)
#caffe_option(USE_GITHUB_MKLDNN "Download and use MKL-DNN available on github" OFF)
 
# ---[ Dependencies
include(cmake/MKLDNN.cmake)
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)

  # Linker flags.
  if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel") 
    # GCC specific flags. ICC is compatible with them.
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -z noexecstack -z relro -z now")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z noexecstack -z relro -z now")
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") 
    # In Clang, -z flags are not compatible, they need to be passed to linker via -Wl.
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
  endif()
  
  # Compiler flags.
  if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") 
    # GCC specific flags.
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.9)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector-strong")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector")
    endif()
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") 
    # Clang is compatbile with some of the flags.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector")
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" ) 
    # Same as above, with exception that ICC compilation crashes with -fPIE option, even 
    # though it uses -pie linker option that require -fPIE during compilation. Checksec
    # shows that it generates correct PIE anyway if only -pie is provided.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")
  endif()
  
  # Generic flags.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-operator-names -Wformat -Wformat-security -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  # Dot not forward c++11 flag to GPU beucause it is not supported
  set( CUDA_PROPAGATE_HOST_FLAGS OFF )
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
endif()

caffe_set_caffe_link()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${Caffe_INCLUDE_DIR} ${PROJECT_BINARY_DIR})
include_directories(BEFORE src/gtest/include src/gmock/include) # This is needed for gtest.
include_directories($ENV{MPI_HOME}/include)

# ---[ Subdirectories
#add_subdirectory(src/gtest)
set(BUILD_SHARED_LIBS off)
add_subdirectory(src/gmock)
set(BUILD_SHARED_LIBS on)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)
add_subdirectory(scripts/SystemTap)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if(BUILD_python)
  add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()

set(CMAKE_CXX_STANDARD 11)
set(TEST_GROUPS test/test_groups.cpp src/caffe/parallel/groups.cpp include/caffe/parallel/groups.hpp)
add_executable(test_groups ${TEST_GROUPS})

