cmake_minimum_required(VERSION 3.4.1)

set(cl_stub_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(cl_stub_SRC "${cl_stub_DIR}/cl_stub-master")
set(cl_stub_build_DIR ${PROJECT_BINARY_DIR}/cl_stub_outputs)
file(MAKE_DIRECTORY ${cl_stub_build_DIR})
add_subdirectory(${cl_stub_SRC} ${cl_stub_build_DIR})
add_library(libcl_stub SHARED IMPORTED)
set_target_properties(libcl_stub PROPERTIES IMPORTED_LOCATION ${cl_stub_build_DIR}/${ANDROID_ABI}/libcl_stub.so)
#Qualcomm Math lib
add_library( lib_QML  SHARED   IMPORTED )
set_target_properties(lib_QML PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libQML-1.0.0.so)
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/QML/include/ )

add_library(OpenCLDemo SHARED
            OpenCLDemo.cpp
            OpenCLGEMM.cpp
            openCLNR.cpp
            refNR.cpp)

target_include_directories(OpenCLDemo PRIVATE
                           ${cl_stub_SRC}/include)

target_link_libraries(OpenCLDemo
                      android
                      libcl_stub
                      lib_QML
                      log)