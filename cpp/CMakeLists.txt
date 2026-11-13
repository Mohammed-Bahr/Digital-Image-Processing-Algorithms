cmake_minimum_required(VERSION 3.16)
project(diplab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Core + one source file per algorithm.
file(GLOB DIPLAB_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/foundations/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/point_processing/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/noise/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/smoothing/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/sharpening/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/frequency/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/features/*.cpp")

add_library(diplab_lib STATIC ${DIPLAB_SOURCES})
target_include_directories(diplab_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(diplab_lib PRIVATE -Wall -Wextra)
endif()

add_executable(diplab src/cli.cpp)
target_link_libraries(diplab PRIVATE diplab_lib)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(diplab PRIVATE -Wall -Wextra)
endif()

add_executable(diplab_selftest tests/selftest.cpp)
target_link_libraries(diplab_selftest PRIVATE diplab_lib)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(diplab_selftest PRIVATE -Wall -Wextra)
endif()

enable_testing()
add_test(NAME selftest COMMAND diplab_selftest)

# --------------------------------------------------------------------------
# Optional raylib GUI. The algorithm library above is not modified by this.
# --------------------------------------------------------------------------
option(DIPLAB_BUILD_GUI "Build the raylib desktop GUI" ON)
if(DIPLAB_BUILD_GUI)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(RAYLIB QUIET IMPORTED_TARGET raylib)
  endif()

  if(TARGET PkgConfig::RAYLIB)
    add_executable(diplab_gui
      gui/main_gui.cpp
      gui/app.cpp
      gui/theme.cpp)
    target_link_libraries(diplab_gui PRIVATE diplab_lib PkgConfig::RAYLIB)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(diplab_gui PRIVATE -Wall -Wextra)
    endif()
    message(STATUS "raylib GUI: enabled (diplab_gui)")
  else()
    message(WARNING "raylib not found - the GUI target (diplab_gui) will be skipped")
  endif()
endif()
