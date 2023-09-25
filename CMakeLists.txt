cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 20)

set(CLANG_PATH "/home/linuxbrew/.linuxbrew/bin/clang++")
# set(CLANG_PATH "/usr/bin/clang++")

set(CLANG_PLUGIN_PATH "/home/linuxbrew/.linuxbrew/Cellar/enzyme/0.0.83/lib/ClangEnzyme-16.so")

set(CMAKE_CXX_COMPILER "${CLANG_PATH}")

project(MALAProject)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(MALA
    main.cpp
)

add_subdirectory(matplotplusplus)

target_link_libraries(MALA PUBLIC matplot)

target_compile_options(MALA PRIVATE
    -fplugin=${CLANG_PLUGIN_PATH}
)
