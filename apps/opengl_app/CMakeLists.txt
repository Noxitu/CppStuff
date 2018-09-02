cmake_minimum_required (VERSION 3.8)
project(opengl_app)

find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

add_executable(opengl_app
    "main.cpp")

add_executable(Noxitu::opengl_app ALIAS opengl_app)

target_link_libraries(opengl_app
    PRIVATE
        GLUT::GLUT
        GLEW::GLEW
)

add_custom_command(TARGET opengl_app POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy "${GLUT_glut_LIBRARY}/../../../bin/Release/freeglut.dll" "$<TARGET_FILE_DIR:opengl_app>"
    COMMAND "${CMAKE_COMMAND}" -E copy "${GLEW_LIBRARY}/../../../bin/Release/glew32.dll" "$<TARGET_FILE_DIR:opengl_app>"
    )