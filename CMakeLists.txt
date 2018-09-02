cmake_minimum_required (VERSION 3.8)

project(Noxitu)

set(GLUT_INCLUDE_DIR "D:/Sources/3rdparties/freeglut-3.0.0/include")
set(GLUT_glut_LIBRARY "D:/Sources/3rdparties/freeglut-3.0.0/build/lib/Release/freeglut.lib")

set(GLEW_INCLUDE_DIR "D:/Sources/3rdparties/glew-2.1.0/include")
set(GLEW_LIBRARY "D:/Sources/3rdparties/glew-2.1.0/build/build/lib/Release/glew32.lib")


add_subdirectory(apps/opengl_app)
add_subdirectory(apps/cpu_app)

add_custom_target(run
    COMMAND "$<TARGET_FILE:Noxitu::opengl_app>")

    add_custom_target(run2
    COMMAND "$<TARGET_FILE:Noxitu::cpu_app>")