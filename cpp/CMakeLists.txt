cmake_minimum_required(VERSION 3.10)
project(main)

SET(CMAKE_BUILD_TYPE "Debug") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# set(CMAKE_CXX_FLAGS "-O3")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")
 
include_directories(${PROJECT_SOURCE_DIR}/include)

# use libbmrt libbmlib
find_package(libsophon REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
link_directories(${LIBSOPHON_LIB_DIRS})

find_package(Armadillo REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

set(CTCDECODE_CPP_DIR ${PROJECT_SOURCE_DIR}/ctcdecode-cpp)
include_directories(${CTCDECODE_CPP_DIR}/include)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILES)
add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES} ${SNDFILE_LIBRARIES} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so} ${CTCDECODE_CPP_DIR}/build/libctcdecode.so)
target_include_directories(${PROJECT_NAME} PRIVATE ${SNDFILE_INCLUDE_DIRS})