cmake_minimum_required(VERSION 3.8)

project(Yolo-Op
        VERSION 0.0.0)

add_library(${PROJECT_NAME} STATIC
  "src/noxitu/yolo/fast_convolution.cpp"
  "src/noxitu/yolo/amp_convolution.cpp"
)

target_link_libraries(${PROJECT_NAME}
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

find_package(benchmark REQUIRED)

add_executable(${PROJECT_NAME}_benchmark
  "benchmark/BM_Convolution.cpp"
)

target_link_libraries(${PROJECT_NAME}_benchmark
  PRIVATE
    Noxitu::${PROJECT_NAME}
    benchmark::benchmark
)

find_package(GTest REQUIRED CONFIG)

add_executable(${PROJECT_NAME}_unit_tests
  "unit_tests/Test_Convolution.cpp"
)

target_link_libraries(${PROJECT_NAME}_unit_tests
  PRIVATE
    Noxitu::${PROJECT_NAME}
    GTest::gmock_main
)