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

option(BPNN_TUI "Build the FTXUI live-training terminal dashboard" ON)

if(BPNN_TUI)
    include(FetchContent)
    FetchContent_Declare(
        ftxui
        URL https://github.com/ArthurSonzogni/FTXUI/archive/refs/tags/v5.0.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ftxui)
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

file(GLOB_RECURSE BPNN_SOURCES
    RELATIVE "${CMAKE_SOURCE_DIR}"
    src/*.cpp
    src/*.h
)


add_executable(backpropnn ${BPNN_SOURCES})

if(BPNN_TUI)
    target_compile_definitions(backpropnn PRIVATE BPNN_TUI=1)
    target_link_libraries(backpropnn PRIVATE ftxui::dom ftxui::screen)
endif()