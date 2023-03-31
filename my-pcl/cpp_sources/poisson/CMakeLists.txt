cmake_minimum_required(VERSION 3.20)
project(PoissonReconLib)

option(BUILD_SHARED_LIB "Builds PoissonRecon as a shared library artifact" ON)
option(BUILD_EXECUTABLE "Builds PoissonRecon as a command-line application" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

if (MSVC)
    string(JOIN " " CMAKE_CXX_FLAGS_RELEASE
        ${CMAKE_CXX_FLAGS_RELEASE}
        -DRELEASE
        -DNOMINMAX
        -DWIN32_LEAN_AND_MEAN
        -D_CRT_SECURE_NO_DEPRECATE
        -D_CRT_SECURE_NO_WARNINGS
        /bigobj
    )
else()
    string(JOIN " " CMAKE_CXX_FLAGS_RELEASE
        ${CMAKE_CXX_FLAGS_RELEASE}
        -DRELEASE
        -Ofast
        -funroll-loops
        -ffast-math
        -Wno-deprecated
        -Wno-invalid-offsetof
        -Wno-dangling-else
    )
endif()

string(JOIN " " CMAKE_EXE_LINKER_FLAGS_RELEASE
    ${CMAKE_EXE_LINKER_FLAGS_RELEASE}
    ${CMAKE_CXX_FLAGS_RELEASE}
)

file(GLOB_RECURSE SRCS_ZLIB "ZLIB/*.c")
add_library(ZLIB STATIC ${SRCS_ZLIB})

file(GLOB_RECURSE SRCS_PNG "PNG/*.c")
add_library(PNG STATIC ${SRCS_PNG})

file(GLOB_RECURSE SRCS_JPEG "JPEG/*.cpp")
add_library(JPEG STATIC ${SRCS_JPEG})

include_directories(${CMAKE_CURRENT_LIST_DIR} JPEG ZLIB PNG)

if (BUILD_SHARED_LIB)
    message(STATUS "PoissonReconLib: Building shared library")
    add_library(PoissonReconLib SHARED Src/PoissonReconLib.h Src/PoissonReconLib.cpp)
    target_link_libraries(PoissonReconLib PRIVATE PNG ZLIB JPEG Threads::Threads)

    find_package(OpenMP REQUIRED COMPONENTS C CXX)
    target_link_libraries(PoissonReconLib PRIVATE OpenMP::OpenMP_CXX)

    install(
        TARGETS PoissonReconLib
        EXPORT PoissonReconLibTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
    install(
        EXPORT PoissonReconLibTargets
        NAMESPACE PoissonReconLib::
        DESTINATION lib/cmake/PoissonReconLib
    )

    set(INCLUDE_INSTALL_DIR include)
    install(FILES Src/PoissonReconLib.h DESTINATION ${INCLUDE_INSTALL_DIR})

    # CMake Package Config Files
    include(CMakePackageConfigHelpers)
    configure_package_config_file(${PROJECT_NAME}Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
        PATH_VARS INCLUDE_INSTALL_DIR
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        VERSION 13.8
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION lib/cmake/${PROJECT_NAME}
    )
    export(
        EXPORT PoissonReconLibTargets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake
        NAMESPACE PoissonReconLib::
    )
endif()

if (BUILD_EXECUTABLE)
    message(STATUS "PoissonRecon: Building executable")
    add_executable(PoissonReconApp Src/PoissonRecon.cpp)
    target_link_libraries(PoissonReconApp PRIVATE PNG ZLIB JPEG Threads::Threads)
    install(TARGETS PoissonReconApp)
endif()
