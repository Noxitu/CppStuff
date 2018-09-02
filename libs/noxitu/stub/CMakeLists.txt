cmake_minimum_required(VERSION 3.8)

project(Shared
        VERSION 1.0.0)

add_library(${PROJECT_NAME} SHARED
  main.cpp
)

target_link_libraries(${PROJECT_NAME}
  Noxitu::Static
  )

set(INSTALL_PATH "shared/$<CONFIG>/")

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}
        DESTINATION "${INSTALL_PATH}/")


install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> 
        DESTINATION "${INSTALL_PATH}/"
        OPTIONAL)

install(FILES $<TARGET_FILE:${PROJECT_NAME}> 
        DESTINATION "apps/$<CONFIG>/")

install(EXPORT ${PROJECT_NAME}
        NAMESPACE Noxitu::
        DESTINATION "shared/")


add_library(Noxitu::${PROJECT_NAME}
            ALIAS ${PROJECT_NAME})