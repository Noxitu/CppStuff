cmake_minimum_required (VERSION 3.8)

project(Noxitu)

add_compile_options(/MT)

#set(GLUT_INCLUDE_DIR "D:/Sources/3rdparties/freeglut-3.0.0/include")
#set(GLUT_glut_LIBRARY "D:/Sources/3rdparties/freeglut-3.0.0/build/lib/Release/freeglut.lib")

#set(GLEW_INCLUDE_DIR "D:/Sources/3rdparties/glew-2.1.0/include")
#set(GLEW_LIBRARY "D:/Sources/3rdparties/glew-2.1.0/build/build/lib/Release/glew32.lib")

#set(OpenCV_DIR "C:/Users/Grzegorz/Downloads/opencv/build" CACHE filepath "")
set(CMAKE_PREFIX_PATH "D:/Sources/3rdparties/ninja-install/all" CACHE filepath "")

#add_subdirectory(libs/core/context)
#add_subdirectory(apps/context_app)
#add_custom_target(run COMMAND "$<TARGET_FILE:Noxitu::context_app>")

#add_subdirectory(libs/core/executor)
#add_custom_target(run COMMAND "$<TARGET_FILE:Noxitu::Executor>")

add_subdirectory(libs/yolo/common)
add_subdirectory(libs/yolo/cpu_op)
add_subdirectory(libs/yolo/cpu)
add_subdirectory(apps/custom_yolo_app)

#add_subdirectory(apps/amp_app)
#add_custom_target(run COMMAND "$<TARGET_FILE:Noxitu::amp_app>")

#add_subdirectory(libs/core/collections)
#add_custom_target(run COMMAND "$<TARGET_FILE:Noxitu::collections>")

#add_subdirectory(apps/opengl_app)
#add_subdirectory(apps/cpu_app)


#add_custom_target(run COMMAND "$<TARGET_FILE:Noxitu::opengl_app>")
#add_custom_target(run2 COMMAND "$<TARGET_FILE:Noxitu::cpu_app>")