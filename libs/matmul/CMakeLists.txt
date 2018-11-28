cmake_minimum_required(VERSION 3.8)

project(MatMul 
        VERSION 0.0.0)

find_package(OpenCV REQUIRED core)
find_package(benchmark REQUIRED)
find_package(GTest REQUIRED CONFIG)

add_library(${PROJECT_NAME} STATIC
  "src/noxitu/matmul/amp.cpp"
  "src/noxitu/matmul/cpu.cpp"
  "src/noxitu/matmul/opencv.cpp"
  "src/noxitu/matmul/utils.cpp"
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    opencv_core
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

add_custom_command(OUTPUT "opencv_core344.dll"
    COMMAND "${CMAKE_COMMAND}" -E copy 
      "${OpenCV_LIB_PATH}/../bin/opencv_core344.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>")

add_executable(${PROJECT_NAME}_benchmark
  "opencv_core344.dll"
  "benchmark/BM_MatMul.cpp"
)

target_link_libraries(${PROJECT_NAME}_benchmark
  PRIVATE
    Noxitu::${PROJECT_NAME}
    benchmark::benchmark
)

add_executable(${PROJECT_NAME}_unit_tests
  "opencv_core344.dll"
  "unit_tests/Test_MatMul.cpp"
)

target_link_libraries(${PROJECT_NAME}_unit_tests
  PRIVATE
    Noxitu::${PROJECT_NAME}
    GTest::gmock_main
)