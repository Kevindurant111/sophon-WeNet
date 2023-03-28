cmake_minimum_required(VERSION 3.10)
project(main)

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_BUILD_TYPE "Debug") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# set(CMAKE_CXX_FLAGS "-O3")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message("cur dir: ${PROJECT_SOURCE_DIR}")

find_package(kenlm REQUIRED)
include_directories(/usr/local/include/kenlm)

include_directories(${PROJECT_SOURCE_DIR}/include)

# use libbmrt libbmlib
find_package(libsophon REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
link_directories(${LIBSOPHON_LIB_DIRS})

find_package(Armadillo REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(/usr/local/include/fst)

set(KENLM_MAX_ORDER 6 CACHE STRING "Maximum supported ngram order")
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILES)
add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES} ${SNDFILE_LIBRARIES} ${the_libbmlib.so} ${the_libbmrt.so} ${kenlm_LIBRARIES} pthread)
target_include_directories(${PROJECT_NAME} PRIVATE ${SNDFILE_INCLUDE_DIRS} ${/usr/local/include/fst})
target_compile_definitions(main PUBLIC -DKENLM_MAX_ORDER=${KENLM_MAX_ORDER})
# ${libfst.so} ${libfstfar.so} ${libfstfarscript.so} ${libfstscript.so}



