cmake_minimum_required (VERSION 3.8)
project(custom_yolo_app
  LANGUAGES CXX)

add_executable(custom_yolo_app
    "main.cpp")

add_executable(Noxitu::custom_yolo_app ALIAS custom_yolo_app)

target_link_libraries(custom_yolo_app
    PRIVATE
        Noxitu::Yolo-Cpu
)

target_compile_options(custom_yolo_app 
    PRIVATE 
      $<$<BOOL:${MSVC}>:/arch:AVX2>
)

find_package(OpenCV REQUIRED core)

add_custom_command(TARGET custom_yolo_app POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy "${OpenCV_LIB_PATH}/../bin/opencv_world310.dll" "$<TARGET_FILE_DIR:custom_yolo_app>"
    )