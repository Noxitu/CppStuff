cmake_minimum_required (VERSION 3.8)
project(amp_app)

add_executable(${PROJECT_NAME}
    "main.cpp")

target_compile_options(${PROJECT_NAME}
    PUBLIC 
      $<$<BOOL:${MSVC}>:/openmp>
      $<$<BOOL:${MSVC}>:/arch:AVX2>
)

add_executable(Noxitu::${PROJECT_NAME} ALIAS ${PROJECT_NAME})