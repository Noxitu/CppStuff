cmake_minimum_required (VERSION 3.8)
project(cpu_app)

add_executable(cpu_app
    "main.cpp")

target_compile_options(cpu_app PRIVATE /openmp)

add_executable(Noxitu::cpu_app ALIAS cpu_app)