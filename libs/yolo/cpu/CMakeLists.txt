cmake_minimum_required(VERSION 3.8)

project(Yolo-Cpu
        VERSION 0.0.0)

add_library(${PROJECT_NAME} STATIC
  "src/noxitu/yolo/cpu/fast_convolution.cpp"
  "src/noxitu/yolo/cpu/Network.cpp"
  "src/noxitu/yolo/cpu/NetworkBuilder.cpp"
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC 
    Noxitu::Yolo-Common
)

target_compile_options(${PROJECT_NAME}
    PUBLIC 
      $<$<BOOL:${MSVC}>:/openmp>
      $<$<BOOL:${MSVC}>:/arch:AVX2>
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
)

add_library(Noxitu::${PROJECT_NAME}
            ALIAS ${PROJECT_NAME})