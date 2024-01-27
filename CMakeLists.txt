cmake_minimum_required(VERSION 3.20)

project(BPNN)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wold-style-cast
    -Winit-self
    -Wno-unused
    -Wshadow
    -Wno-parentheses
    -Wlogical-op
    -Wredundant-decls
    -Wcast-align
    -Wsign-promo
    -Wmissing-include-dirs
    -Woverloaded-virtual
    -Wctor-dtor-privacy
)

include_directories(${PROJECT_SOURCE_DIR}/src)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

file(GLOB_RECURSE BPNN_SOURCES
    RELATIVE "${CMAKE_SOURCE_DIR}"
    src/*.cpp
    src/*.h
)


add_executable(backpropnn ${BPNN_SOURCES})
