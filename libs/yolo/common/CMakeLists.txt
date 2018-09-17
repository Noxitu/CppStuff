cmake_minimum_required(VERSION 3.8)

project(Yolo-Common
        VERSION 0.0.0)

find_package(OpenCV REQUIRED core)
set_target_properties(opencv_core PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}") # why?

add_library(${PROJECT_NAME} STATIC
  "src/noxitu/yolo/common/ConfigurationEntry.cpp"
  "src/noxitu/yolo/common/NetworkBuilder.cpp"
  "src/noxitu/yolo/common/NetworkConfiguration.cpp"
  "src/noxitu/yolo/common/Weights.cpp"
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC 
    opencv_core
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
)

add_library(Noxitu::${PROJECT_NAME}
            ALIAS ${PROJECT_NAME})