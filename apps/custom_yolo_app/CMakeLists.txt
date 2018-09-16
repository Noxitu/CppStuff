cmake_minimum_required (VERSION 3.8)
project(custom_yolo_app
  LANGUAGES CXX)

find_package(OpenCV REQUIRED core)
set_target_properties(opencv_core PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}") # why?

add_executable(custom_yolo_app
    "main.cpp")

add_executable(Noxitu::custom_yolo_app ALIAS custom_yolo_app)

target_link_libraries(custom_yolo_app
    PRIVATE
        opencv_core
)

target_compile_options(custom_yolo_app 
    PRIVATE 
      $<$<BOOL:${MSVC}>:/arch:AVX2>
)

add_custom_command(TARGET custom_yolo_app POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy "${OpenCV_LIB_PATH}/../bin/opencv_world310.dll" "$<TARGET_FILE_DIR:custom_yolo_app>"
    )