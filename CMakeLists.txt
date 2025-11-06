cmake_minimum_required(VERSION 3.16)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)

add_executable(MyProject src/main.cpp)
target_link_libraries(MyProject PRIVATE ${OpenCV_LIBS} glfw GLEW::GLEW)
target_include_directories(MyProject PRIVATE ${OpenCV_INCLUDE_DIRS})

# ---- Runtime assets: copy shaders to the binary dir (Debug/Release) ----
add_custom_command(TARGET MyProject POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:MyProject>/shaders"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/shaders/basic.vert"
          "${CMAKE_SOURCE_DIR}/shaders/basic.frag"
          "$<TARGET_FILE_DIR:MyProject>/shaders"
)

# ---- Make VS run from the binary dir (so relative paths resolve) ----
set_property(TARGET MyProject PROPERTY
  VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:MyProject>")
