cmake_minimum_required(VERSION 3.8)

project(Executor
        VERSION 1.0.0)

#add_library
add_executable(${PROJECT_NAME}
  "src/main.cpp"
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
)

#add_library(Noxitu::${PROJECT_NAME}
add_executable(Noxitu::${PROJECT_NAME}
            ALIAS ${PROJECT_NAME})